from pycocotools.coco import COCO
from PIL import Image
import os
import tqdm
import cv2
import imgviz
import numpy as np
import spectral 
import matplotlib.pyplot as plt

def save_colored_mask(save_path, mask):
    """保存调色板彩色图"""
    lbl_pil = Image.fromarray(mask.astype(np.uint8), mode='P')
    colormap = imgviz.label_colormap(80)
    lbl_pil.putpalette(colormap.flatten())
    lbl_pil.save(save_path)


coco_root = r'E:\my_project\hyperspectralData\medician\leaf\drawRigion'
annotation_file = r'E:\my_project\hyperspectralData\medician\leaf\drawRigion\runs\labelme2coco\dataset.json'
mask_root = os.path.join(coco_root, 'maskPic')

save_iscrowd = True

coco = COCO(annotation_file)
# print(coco)
catIds = coco.getCatIds()       # 类别ID列表
imgIds = coco.getImgIds()       # 图像ID列表
# print(catIds)
# print(imgIds)
print("catIds len: {}, imgIds len: {}".format(len(catIds), len(imgIds)))

cats = coco.loadCats(catIds)   # 获取类别信息->dict
# print(cats)
names = [cat['name'] for cat in cats]  # 类名称
print(names)

img_cnt = 0
crowd_cnt = 0

for idx, imgId in tqdm.tqdm(enumerate(imgIds), ncols=100):
    if save_iscrowd:
        annIds = coco.getAnnIds(imgIds=imgId)      # 获取该图像上所有的注释id->list
    # else:
    #     annIds = coco.getAnnIds(imgIds=imgId, iscrowd=False)  # 获取该图像的iscrowd==0的注释id
    if len(annIds) > 0:
        # print(imgId)
        image = coco.loadImgs([imgId])[0]
        hyperSpectralDataPath = '../../../'+image['file_name'][:-4]+'/results/'
        hyperSpectralDataDatPath = hyperSpectralDataPath+'REFLECTANCE_'+image['file_name'][:-4]+'.dat'
        hyperSpectralDataHdrPath = hyperSpectralDataPath+'REFLECTANCE_'+image['file_name'][:-4]+'.hdr'
        hyperSpectralImg = spectral.envi.open(hyperSpectralDataHdrPath, hyperSpectralDataDatPath)
        hyperSpectralImgData = hyperSpectralImg.read_bands([i for i in range(204)])
        # print(hyperSpectralImgData.shape)
        # spectral.envi.save_image('pcimage.hdr', hyperSpectralImgData, dtype=np.float32)
        # print(help(hyperSpectralImg))
        # hyperSpectralImg = spectral.envi.open('pcimage.hdr', 'pcimage.img')
        # print(hyperSpectralImg)
        # print(help(spectral.envi.save_image))
        # view = spectral.imshow(hyperSpectralImg, (29, 19, 9))
        # print(hyperSpectralImg[300, 190])
        # cv2.waitKey()
        # for i in range(0, hyperSpectralImg.shape[2]):
        #     p = hyperSpectralImg.read_band(i)
        #     # print(p)
        #     # hyperSpectralImg[:,:,i] = p
        #     plt.imshow(p)
        #     plt.show()
        
        # print(os.listdir(hyperSpectralDataPath))
        # print(image['file_name'][:-4])
        # break
        ## ['coco_url', 'flickr_url', 'date_captured', 'license', 'width', 'height', 'file_name', 'id']
        h, w = image['height'], image['width']
        gt_name = image['file_name'].replace('.jpg', '.png')
        # print(gt_name)
        # gt = np.zeros((h, w), dtype=np.uint8)
        anns = coco.loadAnns(annIds)    # 获取所有注释信息
        # print(anns)
        # break
        has_crowd_flag = 0
        save_flag = 0
        for ann_idx, ann in enumerate(anns):
            hyperSpectralImgDataSplit = np.zeros(hyperSpectralImgData.shape)
            gt = np.zeros((h, w), dtype=np.uint8)
            cat = coco.loadCats([ann['category_id']])[0]
            cat = cat['name']
            cat = names.index(cat) + 1   # re-map
            # print(cat)
            # break
            if not ann['iscrowd']:  # iscrowd==0
                segs = ann['segmentation']
                for seg in segs:
                    seg = np.array(seg).reshape(-1, 2)     # [n_points, 2]
                    # cv2.fillPoly(gt, seg.astype(np.int32)[np.newaxis, :, :], int(cat))
                    cv2.fillPoly(gt, seg.astype(np.int32)[np.newaxis, :, :], 1)
                    
                    # print('1')
            # elif save_iscrowd:
            #     has_crowd_flag = 1
            #     rle = ann['segmentation']['counts']
            #     assert sum(rle) == ann['segmentation']['size'][0] * ann['segmentation']['size'][1]
            #     mask = coco.annToMask(ann)
            #     unique_label = list(np.unique(mask))
            #     assert len(unique_label) == 2 and 1 in unique_label and 0 in unique_label
            #     gt = gt * (1 - mask) + mask * 255   # 这部分填充255
            #     print('2')
            # print(ann_idx)
            # print(gt_name)
            gt_name_1 = gt_name[0:-4]+'_'+str(ann_idx)+'.png'
            # print(gt_name)
            save_path = os.path.join(mask_root, gt_name_1)
            # print(save_path)
            # print(gt)
            gt = cv2.rotate(gt, cv2.ROTATE_90_COUNTERCLOCKWISE) # ROTATE_90_CLOCKWISE ROTATE_180
            # hyperSpectralImgDataSplit = np.logical_and(hyperSpectralImgData, gt)
            gt = np.expand_dims(gt, 2)
            gt = gt.repeat(204, axis=2)
            # print(gt.shape)
            hyperSpectralImgDataSplit = hyperSpectralImgData * gt
            # print(hyperSpectralImgDataSplit[300, 190])
            position = ann['bbox']
            spectral.envi.save_image('./hyperspectralDataSplit/'+gt_name[0:-4]+'_'+str(ann_idx)+'.hdr', hyperSpectralImgDataSplit[511-int(position[0]+position[2]):511-int(position[0]),int(position[1]):int(position[1]+position[3])], dtype=np.float32)
            # print(position)
            # print(ann['segmentation'])
            # print(512-int(position[0]+position[2]), int(position[0]))
            # view = spectral.imshow(hyperSpectralImgDataSplit[511-int(position[0]+position[2]):511-int(position[0]),int(position[1]):int(position[1]+position[3])], (29, 19, 9))
            # view = spectral.imshow(hyperSpectralImg[511-int(position[0]+position[2]):511-int(position[0]),int(position[1]):int(position[1]+position[3])], (29, 19, 9))
            # view = spectral.imshow(hyperSpectralImg[511-int(position[0]+position[2])+20:511-int(position[0]),int(position[1]):int(position[1]+position[3])], (29, 19, 9))
            # plt.pause(100000)
            # cv2.imwrite(save_path, gt)
        # save_path = os.path.join(mask_root, gt_name)
        # cv2.imwrite(save_path, gt)
        img_cnt += 1
        if has_crowd_flag:
            crowd_cnt += 1

        if idx % 100 == 0:
            print('Processed {}/{} images.'.format(idx, len(imgIds)))

print('crowd/all = {}/{}'.format(crowd_cnt, img_cnt))