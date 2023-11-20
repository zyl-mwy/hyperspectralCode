# %pylab

import matplotlib.pyplot as plt
import spectral as spy
import scipy
import os 
from spectral import view_cube
# pip install wxPython
# pip install pyOpenGL
# from pylab import *

# spy.settings.WX_GL_DEPTH_SIZE = 16

# PCA主成分分析
def pca_dr(src):
    pc = spy.principal_components(src)
    # print(pc)
    pc_98 = pc.reduce(fraction=0.9999)  # 保留98%的特征值
    print(len(pc_98.eigenvalues))  # 剩下的特征值数量
    # print(pc.cov.shape)
    # spy.imshow(data=pc.cov, title="pc_cov")
    img_pc = pc_98.transform(src)  # 把数据转换到主成分空间
    # print(img_pc.shape)
    # plt.imshow(src[:, :, :3])
    # plt.show()
    # plt.imshow(img_pc[:, :, :3])
    # plt.show()
    # spy.imshow(img_pc[:, :, :3], stretch_all=True)  # 前三个主成分显示
    for i in range(img_pc.shape[2]):
        plt.subplot(4, 5, i+1)
        plt.imshow(img_pc[:, :, i], cmap='gray')
        plt.axis('off')
    plt.show()
    return img_pc

if __name__ == "__main__":
    osPath = r'E:\my_project\hyperspectralData\medician\leaf\drawRigion\runs\labelme2coco\hyperspectralDataSplit'
    hyperSpectalFile = spy.envi.open(os.path.join(osPath, '2023-10-12_009_0.hdr'), os.path.join(osPath, '2023-10-12_009_0.img')).read_bands([i for i in range(204)])
    spy.view_cube(hyperSpectalFile, bands=[29, 19, 9])
    # pca_dr(hyperSpectalFile)
    for i in range(204):
        plt.subplot(12, 17, i+1)
        plt.imshow(hyperSpectalFile[:, :, i], cmap='gray')
        plt.axis('off')
    plt.show()
    pca_dr(hyperSpectalFile)
    
        