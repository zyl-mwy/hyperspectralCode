import matplotlib.pyplot as plt
import spectral
import scipy
import os
if __name__ == "__main__":
    osPath = r'E:\my_project\hyperspectralData\medician\leaf\drawRigion\runs\labelme2coco\hyperspectralDataSplit'
    hyperSpectalFile = spectral.envi.open(os.path.join(osPath, '2023-10-12_009_0.hdr'), os.path.join(osPath, '2023-10-12_009_0.img')).read_bands([i for i in range(204)])
    # hyperSpectalFile = hyperSpectalFile.sum(axis=0)
    # hyperSpectalFile = hyperSpectalFile.sum(axis=0)
    # print(hyperSpectalFile.shape)
    hyperSpectalFile = hyperSpectalFile[43, 26, :]
    y_smooth = scipy.signal.savgol_filter(hyperSpectalFile,10,3) 
    plt.rcParams ['font.sans-serif'] = ['Simhei'] 
    # plt.subplot(121)
    plt.title('滤波前')
    plt.plot(hyperSpectalFile[10:-10])
    plt.show()
    # plt.subplot(122)
    plt.title('滤波后')
    plt.plot(y_smooth[10:-10])
    plt.show()
    # print(hyperSpectalFile.shape)
    