# https://blog.csdn.net/qq_42672745/article/details/113426995
import matplotlib.pyplot as plt
from scipy.io import loadmat
import spectral as spy


# PCA主成分分析
def pca_dr(src):
    pc = spy.principal_components(src)
    pc_98 = pc.reduce(fraction=0.98)  # 保留98%的特征值
    print(len(pc_98.eigenvalues))  # 剩下的特征值数量
    spy.imshow(data=pc.cov, title="pc_cov")
    img_pc = pc_98.transform(input_image)  # 把数据转换到主成分空间
    spy.imshow(img_pc[:, :, :3], stretch_all=True)  # 前三个主成分显示
    return img_pc


# LDA线性判别
def lda_dr(src, gt):
    classes = spy.create_training_classes(src, gt)
    fld = spy.linear_discriminant(classes)
    print(len(fld.eigenvalues))
    img_fld = fld.transform(src)
    spy.imshow(img_fld[:, :, :3])
    return img_fld


input_image = loadmat('D:/Hyper/Indian_pines_corrected.mat')['indian_pines_corrected']
gt = loadmat("D:/Hyper/Indian_pines_gt.mat")['indian_pines_gt']

pca_dr(input_image)
lda_dr(input_image, gt)

plt.pause(60)
