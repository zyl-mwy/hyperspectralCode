import spectral
import os
import tqdm
import cv2 as cv
import numpy as np
hdrFileName = None
imgFileName = None
osPath = r'E:\my_project\hyperspectralData\medician\leaf\drawRigion\runs\labelme2coco\hyperspectralDataSplit'
savePath = r'E:\my_project\hyperspectralData\medician\leaf\drawRigion\maskPicNormal\\'
findMaxMin = {'WidthMax': 0, 'WidthMin': 100, 'HeightMax': 0, 'HeightMin': 100}
regionShape = [16, 16]
for i, fileName in tqdm.tqdm(enumerate(os.listdir(osPath))):
    if i%2 == 0:
        hdrFileName = fileName
        # print(i+1, fileName)
    else:
        imgFileName = fileName
        # print(hdrFileName, imgFileName)
        hyperSpectalFile = spectral.envi.open(os.path.join(osPath, hdrFileName), os.path.join(osPath, imgFileName))
        hyperSpectalFileShape = hyperSpectalFile.shape
        hyperSpectralImgData = hyperSpectalFile.read_bands([i for i in range(204)])[(hyperSpectalFileShape[0]-regionShape[0])//2:(hyperSpectalFileShape[0]-regionShape[0])//2+regionShape[0], (hyperSpectalFileShape[1]-regionShape[1])//2:(hyperSpectalFileShape[1]-regionShape[1])//2+regionShape[1]]
        print(hyperSpectralImgData.shape, hyperSpectalFileShape, (hyperSpectalFileShape[0]-regionShape[0])//2, (hyperSpectalFileShape[0]-regionShape[0])//2+regionShape[0], (hyperSpectalFileShape[1]-regionShape[1])//2, (hyperSpectalFileShape[1]-regionShape[1])//2+regionShape[1])
        # view = spectral.imshow(hyperSpectralImgData, (29, 19, 9))
        # cv.waitKey(1000)
        # view = spectral.imshow(hyperSpectalFile, (29, 19, 9))
        # cv.waitKey(1000)
        print(savePath+fileName[0:-4]+'.hdr')
        spectral.envi.save_image(savePath+fileName[0:-4]+'.hdr', hyperSpectralImgData, dtype=np.float32)
        # print(hyperSpectalFileShape)
#         if findMaxMin['WidthMax'] < hyperSpectalFileShape[1]:
#             findMaxMin['WidthMax'] = hyperSpectalFileShape[1]
#         if findMaxMin['WidthMin'] > hyperSpectalFileShape[1]:
#             findMaxMin['WidthMin'] = hyperSpectalFileShape[1]
#         if findMaxMin['HeightMax'] < hyperSpectalFileShape[0]:
#             findMaxMin['HeightMax'] = hyperSpectalFileShape[0]
#         if findMaxMin['HeightMin'] > hyperSpectalFileShape[0]:
#             findMaxMin['HeightMin'] = hyperSpectalFileShape[0]
# print(findMaxMin)