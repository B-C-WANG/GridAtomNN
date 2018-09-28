from VDE.VASPMoleculeFeature import VASP_DataExtract
from AtomGrid import AtomGrid
import numpy as np
from GridConv3D import GridConv3D
import os
import matplotlib.pyplot as plt

plt.switch_backend('agg')

def make_dataset(vasp_dir,atom_case,save=False,max_dataset_number=-1):
    a = VASP_DataExtract(vasp_dir=vasp_dir)

    vasp_dir_name = vasp_dir.split("/")[-1].split('\\')[-1]

    c = a.get_output_as_atom3Dspace()
    e = a.get_energy_info()

    coord, energy, _atom_case = c.generate_data()

    for i in _atom_case:
        if i not in atom_case:
            raise ValueError("Input Atom Case Didn't Contain %s, make sure all atom cases are in atom_case" % _atom_case)
    atom_case = atom_case

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

    all_index = list(range(coord.shape[0]))[: max_dataset_number]
    np.random.shuffle(all_index)
    print(all_index)
    sample_num = len(all_index)
    train_test_ratio = 0.7
    trainX = testX = None
    train_index = all_index[:int(sample_num * train_test_ratio)]
    test_index = all_index[int(sample_num * train_test_ratio):]
    if (save != False):
        a.grid_encode(train_index, "%s_trainX.npy"%vasp_dir_name)
        a.grid_encode(test_index, "%s_testX.npy"%vasp_dir_name)
    else:
        trainX = a.grid_encode(train_index,save=False)
        testX = a.grid_encode(test_index,save=False)

    trainY = []
    testY = []
    for i in train_index:
        trainY.append(energy[i])
    for i in test_index:
        testY.append(energy[i])

    trainY = np.array(trainY)
    testY = np.array(testY)
    if save != False:
        np.save("%s_trainY.npy"%vasp_dir_name,trainY)
        np.save("%s_testY.npy"%vasp_dir_name,testY)
    else :
        return trainX,trainY,testX,testY

def load_dataset(vasp_dir):
    vasp_dir_name = vasp_dir.split("/")[-1].split('\\')[-1]
    return np.load("%s_trainX.npy"%vasp_dir_name),\
           np.load("%s_trainY.npy"%vasp_dir_name), \
           np.load("%s_testX.npy"%vasp_dir_name), \
           np.load("%s_testY.npy"%vasp_dir_name)




if __name__ == '__main__':


    # TODO: 增加交替训练，这很重要！！
    all_true_y = []
    all_pred_y = []

    atom_case = (1,6,8,78)

    np.random.seed(1)

    a = GridConv3D(60, 60, 60, 4) # 四种元素，先训练CHOPt都有的
    a.build()

    all_vasp_dir = VASP_DataExtract.get_all_VASP_dirs_in_dir(os.getcwd()) # 获得当前的VASP文件夹作为训练集
    index = 0
    for vasp_dir in all_vasp_dir:
        index += 1
        print("train for %s" % vasp_dir)
        print("Total Process: %s/%s" % (index,len(all_vasp_dir)))
        # 选择不保存
        trainX, trainY, testX, testY = make_dataset(vasp_dir,save=False,max_dataset_number=50,atom_case=atom_case) # 最多50个sample拿来训练
        # 如果保存可以载入
        #trainX,trainY,testX,testY = load_dataset(vasp_dir)

        a.fit(trainX,trainY,epochs=1,batch_size=1)

    for vasp_dir in all_vasp_dir:
        # 如果之前没有保存，需要重新make
        _, _, testX, testY = make_dataset(vasp_dir,save=False,max_dataset_number=50,atom_case=atom_case)

        #trainX, trainY, testX, testY = load_dataset(vasp_dir)
        pred = a.predict(testX)
        print(pred)
        print(testY)

        all_true_y.extend(list(testY))
        all_pred_y.extend(list(pred))
    plt.plot(all_pred_y,all_true_y,"ro")
    plt.savefig("results.png",dpi=300)








