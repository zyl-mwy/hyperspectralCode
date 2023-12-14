## https://blog.csdn.net/qq_43426078/article/details/124130531
import numpy as np
from matplotlib import pyplot as plt
import pywt
import PIL
import spectral
import cv2 as cv
import os
# img = spectral.envi.open('2023-10-12_009_0.hdr', '2023-10-12_009_0.img')
osPath = r'E:\my_project\hyperspectralData\medician\leaf\drawRigion\runs\labelme2coco\hyperspectralDataSplit'
img = spectral.envi.open(os.path.join(osPath, '2023-10-12_009_0.hdr'), os.path.join(osPath, '2023-10-12_009_0.img')).read_bands([i for i in range(204)])
# img = PIL.Image.open("main1.png") # xxx.tif
# img = np.array(img)[:, :, 0]
# img.imshow()
a = np.zeros([img.shape[0], img.shape[1], 3])
b = np.array(img[:, :, 36:97])
a[:, :, 0] = img[:, :, 13]
a[:, :, 1] = img[:, :, 51]
a[:, :, 2] = img[:, :, 103]
# cv.imshow('haha', a)
# cv.waitKey()
# spectral.imshow(img, (13, 51, 103))
# plt.pause(1000000)
for i in range(204):
    img1 = img[:, :, i]
    LLY,(LHY,HLY,HHY) = pywt.dwt2(img1, 'haar')
    plt.subplot(3, 2, 1)
    plt.axis('off')
    plt.imshow(LLY, cmap="Greys")
    plt.subplot(3, 2, 2)
    plt.axis('off')
    plt.imshow(LHY, cmap="Greys")
    plt.subplot(3, 2, 3)
    plt.axis('off')
    plt.imshow(HLY, cmap="Greys")
    plt.subplot(3, 2, 4)
    plt.axis('off')
    plt.imshow(HHY, cmap="Greys")
    plt.subplot(3, 2, 5)
    plt.axis('off')
    plt.imshow(img1, cmap="Greys")
    plt.show()
