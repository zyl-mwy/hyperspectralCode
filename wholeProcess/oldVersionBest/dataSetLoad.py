import spectral 
import pandas as pd
from torch.utils.data import Dataset
import os
import torch
import numpy as np

class Hyper_IMG(Dataset):

    def __init__(self, root, train='Train', transform = None, target_transform=None):
        super(Hyper_IMG, self).__init__()
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        if self.train=='Train':
            file_annotation = root + '/FinalTrain.xlsx'
            img_folder = root + '/drawRigion/maskPicNormal'
        elif self.train=='Val':
            file_annotation = root + '/FinalVal.xlsx'
            img_folder = root + '/drawRigion/maskPicNormal'
        elif self.train=='Test':
            file_annotation = root + '/FinalTest.xlsx'
            img_folder = root + '/drawRigion/maskPicNormal'
        myAnnotion = pd.read_excel(file_annotation).values
        # assert len(data_dict['images'])==len(data_dict['categories'])
        # print(myAnnotion.shape[0])
        self.num_data = myAnnotion.shape[0]
        self.filenames = []
        self.labels1 = []
        self.labels2 = []
        self.img_folder = img_folder
        # print('---', self.img_folder)
        for i in range(self.num_data):
            self.filenames.append(myAnnotion[i][2])
            self.labels1.append(myAnnotion[i][8])
            self.labels2.append(myAnnotion[i][9])
        # print(self.filenames)
        # print(self.labels1)
        # print(self.labels2)

    def __getitem__(self, index):
        img_name = os.path.join(self.img_folder, self.filenames[index])
        # label1 = torch.FloatTensor([(self.labels1[index]-303.31)/57.96])
        # label2 = torch.FloatTensor([(self.labels2[index]-123.17)/30.72])
        label1 = torch.FloatTensor([self.labels1[index] / 500])
        label2 = torch.FloatTensor([self.labels2[index] / 500])
        # img = (spectral.envi.open(img_name+'.hdr', img_name+'.img').read_bands([i for i in range(204)])-0.58)/0.39
        img = spectral.envi.open(img_name+'.hdr', img_name+'.img').read_bands([i for i in range(204)]) / 2
        
        # print(img_name)
        # print(img)
        # img = plt.imread(img_name)
        # print(np.mean(img), np.std(img))
        if self.transform is not None:
            img = torch.unsqueeze(self.transform(img).permute(1, 2, 0), dim=0) # 
        # print(img.shape)


        # return img, label
        # print(self.img_folder)
        return img, label1, label2#  + self.filenames[index]

    def __len__(self):
        return self.num_data