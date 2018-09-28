# coding: utf-8

import numpy as np
from VDE.VASPMoleculeFeature import VASP_DataExtract

class AtomGrid():

    '''
    第一种方法：设置通道，每个通道是特定种类的那种原子的距离分之一的和
    格子数目，也就是分辨率是确定的，坐标边界需要人为规定
    然后划出每个格子的坐标，然后来求取3维的通道值
    '''
    def __init__(self,
                 coordinate,
                 atom_case):
        # 注意这里的coordinate是样本数目，原子数目，4，4包括原子序数+xyz
        self.coordinate = coordinate
        self.atom_case = atom_case
        self.n_channel = len(list(set(atom_case)))
        self.channel_index = sorted(list(set(atom_case)))
        print(self.channel_index)
        self.grid_distance = None

    def get_grid_border(self):

        '''
        提供原子坐标范围，人为手动确定格子的范围
        '''
        minX = np.min(self.coordinate[:,:,1])
        maxX = np.max(self.coordinate[:,:,1])

        minY = np.min(self.coordinate[:, :, 2])
        maxY = np.max(self.coordinate[:, :, 2])

        minZ = np.min(self.coordinate[:, :, 3])
        maxZ = np.max(self.coordinate[:, :, 3])

        print("Border X:",minX,maxX)
        print("Border Y:",minY,maxY)
        print("Border Z:",minZ,maxZ)

    def make_grid(self,minX,maxX,minY,
                  maxY,minZ,maxZ,
                  resolutionX,resolutionY,resolutionZ):
        '''
         格点范围以及总共一个维度多少个格子
         按照图像处理的大小，可以尝试200x200x200 x n_atom_case
        '''
        deltaX = (maxX - minX) / resolutionX
        deltaY = (maxY - minY) / resolutionY
        deltaZ = (maxZ - minZ) / resolutionZ

        self.grid_distance = np.zeros(shape=(resolutionX,resolutionY,resolutionZ,3))

        for x in range(resolutionX):
            for y in range(resolutionY):
                for z in range(resolutionZ):
                    self.grid_distance[x,y,z] = np.array([deltaX * x + minX, deltaY * y + minY, deltaZ * z + minZ])

        #print(self.grid_distance)


    def grid_encode_slow(self,sample_index,save_file="encoded_grid.npy"):
        self.total_encoded_grid = []
        count = 0
        for i in sample_index:
            count += 1
            print("Sample Process: ",count/len(sample_index)*100)
            coord = self.coordinate[i,:,:]
            if self.grid_distance is None:
                raise ValueError("Please run make_grid first.")
            shape = self.grid_distance.shape
            self.encoded_grid = np.zeros(shape=(shape[0],shape[1],shape[2],len(self.atom_case)))
            # TODO: 去掉循环，改用numpy广播，或者tf处理
            for x in range(shape[0]):
                print("Process One Sample: ",x/shape[0]*100)
                for y in range(shape[1]):
                    for z in range(shape[2]):
                        for atom_case_index in range(len(self.atom_case)):


                            # 获得原子case相同的矩阵，然后求得距离倒数分之1
                            self.encoded_grid[x,y,z,atom_case_index] = \
                            np.sum(# 然后总的sum一下
                            1/(# 然后分之一
                                np.sqrt(# 然后开方加0.5
                                    np.sum(# 然后在第二个轴上sum
                                        np.square(# 然后平方
                                            # 获得原子序数相差小于0.01的所有原子的坐标，然后减去grid的distance
                                            (coord[(coord[:,0]-self.atom_case[atom_case_index])<0.1][:,1:] - self.grid_distance[x,y,z])
                                        )
                                        ,axis=1)
                                )+0.5
                            )
                            )
            self.total_encoded_grid.append(self.encoded_grid)
            # 由于运行很慢，所以存储一下
        self.total_encoded_grid = np.array(self.total_encoded_grid)
        if save_file != False:
                assert isinstance(save_file,str)
                np.save(save_file,self.total_encoded_grid)
        return self.total_encoded_grid


    def grid_encode(self,sample_index,save=False,save_file="encoded_grid.npy"):
        def g_encode(index,coord,grid_distance,method=0):

            '''
            encode方式要使得对微小变化敏感，编码过后需要在featurePlot.py中查看编码过后的image
            '''
            # 第一种：采用距离+0.5分之1
            if method == 0:
                return 1 / (0.5 + np.sqrt(np.sum(np.square(coord[index, :] - grid_distance), axis=1)))
            # 第二种，采用最近距离+1分之1的平方，这样间隔近的权重大
            elif method == 1:
                return 1 / np.square(1 +
                                     np.sqrt(
                                         np.sum(
                                             np.square(
                                                 coord[index, :] - grid_distance), axis=1)))


        self.total_encoded_grid = []
        count = 0
        for i in sample_index:
            count += 1
            print("Sample Process: ",count/len(sample_index)*100)
            coord = self.coordinate[i,:,:]
            if self.grid_distance is None:
                raise ValueError("Please run make_grid first.")
            shape = self.grid_distance.shape
            self.encoded_grid = np.zeros(shape=(shape[0],shape[1],shape[2],len(self.atom_case)))

            grid_distance = self.grid_distance.reshape(shape[0]*shape[1]*shape[2],-1)

            self.encoded_flatten = []
            for atom_case_index in self.atom_case:
                '''
                对于每种原子
                先求第一个原子的距离+0.5分之一
                然后加上其他原子的
                '''


                atom_coord = coord[abs(coord[:,0]-atom_case_index)<0.1][:,1:]
                # 如果没有这种原子，就加上0，注意是flatten过后的
                if atom_coord.shape[0] == 0:
                    self.encoded_flatten.append(np.zeros(shape=(shape[0]*shape[1]*shape[2])))
                    continue
                first_one = g_encode(0,atom_coord,grid_distance)

                if atom_coord.shape[0] > 1:
                    for atom_index in range(1,atom_coord.shape[0]):
                        first_one += g_encode(atom_index,atom_coord,grid_distance)
                self.encoded_flatten.append(first_one)
            #print(len(self.encoded_flatten))
            #print(self.encoded_flatten[0].shape)
            self.encoded_flatten = np.stack(self.encoded_flatten,axis=1)
            #print(self.encoded_flatten.shape)

            self.encoded_grid = self.encoded_flatten.reshape(shape[0],shape[1],shape[2],-1)


            self.total_encoded_grid.append(self.encoded_grid)
            # 由于运行很慢，所以存储一下
        self.total_encoded_grid = np.array(self.total_encoded_grid)
        if save != False:
                assert isinstance(save_file,str)
                np.save(save_file,self.total_encoded_grid)
        return self.total_encoded_grid

    def debug_encoded_grid(self,file_name="encoded_grid.npy"):
        temp = np.load(file_name)
        print(temp[0,0,0])
        print(temp[0,0,temp.shape[2]-1])
        print(temp[0, temp.shape[1]-1, temp.shape[2]-1])
        print(temp[temp.shape[0]-1, temp.shape[1]-1, temp.shape[2]-1])

    def load_encoded_grid(self,file_name):
        return np.load(file_name)

if __name__ == '__main__':
    a = VASP_DataExtract(vasp_dir="data\Pd_CH_s_fcc_stand.cif")
    c = a.get_output_as_atom3Dspace()
    e = a.get_energy_info()

    coord, energy, atom_case = c.generate_data()

    # 1 得到坐标的atom case的信息，输入Vasp dir
    # print(coord)
    a = AtomGrid(coord, atom_case)

    # 2 得到边界信息，然后根据边界人为确定格子大小和边界
    a.get_grid_border()
    a.make_grid(
        minX=-6,
        maxX=15,
        minY=-6,
        maxY=15,
        minZ=-6,
        maxZ=15,
        resolutionX=60,
        resolutionY=60,
        resolutionZ=60
    )

    # 3 进行编码，给出编码的index，打乱分数据集,然后存储不同的数据集

    all_index = list(range(coord.shape[0]))
    np.random.shuffle(all_index)
    print(all_index)
    sample_num = len(all_index)
    train_test_ratio = 0.7

    train_index = all_index[:int(sample_num * train_test_ratio)]
    test_index = all_index[int(sample_num * train_test_ratio):]

    a.grid_encode(train_index, "trainX.npy")
    a.grid_encode(test_index, "testX.npy")




