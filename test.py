from AtomGrid import *
from VDE.VASPMoleculeFeature import VASP_DataExtract
import numpy as np


def encode_method_test():
    a = VASP_DataExtract(vasp_dir="data\Pd_CH_s_fcc_stand.cif")
    c = a.get_output_as_atom3Dspace()
    e = a.get_energy_info()

    coord, energy, atom_case = c.generate_data()

    a = AtomGrid(coord,atom_case)


    a.get_grid_border()
    a.make_grid(
        minX=-0.5,
        maxX=7,
        minY=-3,
        maxY=7.5,
        minZ=-0.5,
        maxZ=9.5,
        resolutionX=100,
        resolutionY=100,
        resolutionZ=100
    )

    train_index = [1]

    en2 = a.grid_encode(train_index)[0]
    en1 = a.grid_encode1(train_index)[0]

    # en2 方法比en1快很多很多，同时要比较一下编码结果是否一致
    print(en1)
    print("_______________________")
    print(en2)
    print(en1-en2)
    print(np.sum(en1-en2))



if __name__ == '__main__':
    encode_method_test()