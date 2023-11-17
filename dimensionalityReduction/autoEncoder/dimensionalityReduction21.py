# https://zhuanlan.zhihu.com/p/100563675
import torch
import torchvision
from torchvision import datasets,transforms
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

transform = transforms.Compose([
     transforms.ToTensor(),
     #transforms.Lambda(lambda x: x.repeat(3,1,1)), #转化为三通道，这里不合适
     transforms.Normalize(mean=[0.5], std=[0.5])])   # 修改的位置，不能是[0.5,0.5,0.5]，只取一个通道，原书中不正确

dataset_train=datasets.MNIST(root="./data",transform=transform,train=True,download=True)
dataset_test=datasets.MNIST(root="./data", transform=transform,train=False)
train_load=torch.utils.data.DataLoader(dataset=dataset_train,batch_size=4,shuffle=True)
test_load=torch.utils.data.DataLoader(dataset=dataset_test,batch_size=4,shuffle=True)

images,label=next(iter(train_load))
print(images.shape)
images_example=torchvision.utils.make_grid(images)
images_example=images_example.numpy().transpose(1,2,0)
mean=0.5
std=0.5
images_example=images_example*std+mean
plt.imshow(images_example)
plt.show()

noisy_images=images_example+0.5*np.random.randn(*images_example.shape)#images_example.shape前没有*就无法指定地址
noisy_images=np.clip(noisy_images,0.,1.)
plt.imshow(noisy_images)
plt.show()


class AutoEncoder(torch.nn.Module):
    def __init__(self):
        super(AutoEncoder,self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1,64,kernel_size=3,stride=1,padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2,stride=2),
            torch.nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2,stride=2))
        self.decoder = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2,mode="nearest"),
            torch.nn.Conv2d(128,64,kernel_size=3,stride=1,padding=1),
            torch.nn.ReLU(),
            torch.nn.Upsample(scale_factor=2,mode="nearest"),
            torch.nn.Conv2d(64,1,kernel_size=3,stride=1,padding=1))
        
    def forward(self,input):
        output = self.encoder(input)
        output = self.decoder(output)
        return output
    
model = AutoEncoder()

Use_gpu=torch.cuda.is_available() #前面用了device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if Use_gpu:
    model=model.cuda()
print(model)
optimizer=torch.optim.Adam(model.parameters())
loss_f=torch.nn.MSELoss()


epoch_n=5
for epoch in range(epoch_n):
    running_loss=0.0
    
    print("Epoch {}/{}".format(epoch+1,epoch_n))
    print("-"*10)
    
    for data in train_load:
        X_train,_=data
        
        noisy_X_train=X_train+0.5*torch.randn(X_train.shape)
        noisy_X_train=torch.clamp(noisy_X_train, 0., 1.)
        X_train,noisy_X_train=Variable(X_train.cuda()),Variable(noisy_X_train.cuda())

        train_pre=model(noisy_X_train)
        loss=loss_f(train_pre,X_train)
        
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step()
        
        running_loss+=loss.item()
        
    print("Loss is:{:.4f}".format(running_loss/len(dataset_train)))


    data_loader_test=torch.utils.data.DataLoader(dataset=dataset_test,batch_size=4,shuffle=True)
X_test,_=next(iter(data_loader_test))
img1=torchvision.utils.make_grid(X_test)
img1=img1.numpy().transpose(1,2,0)
mean=[0.5,0.5,0.5]
std=[0.5,0.5,0.5]
img1=img1*std+mean
noisy_X_test=img1+0.5*np.random.randn(*img1.shape)
noisy_X_test=np.clip(noisy_X_test,0.,1.)
plt.figure()
plt.imshow(noisy_X_test)


img2=X_test+0.5*torch.randn(*X_test.shape)
img2=torch.clamp(img2,0.,1.)
img2=Variable(img2.cuda())#cpu计算view(-1,28*28)
#X_train,noisy_X_train=Variable(X_train.cuda()),Variable(noisy_X_train.cuda())
test_pred=model(img2)

img_test=test_pred.data.view(-1,1,28,28)
#img_test=img_test.cpu()
img2=torchvision.utils.make_grid(img_test)
img2=img2.cpu()
img2=img2.numpy().transpose(1,2,0)
img2=img2*std+mean
img2=np.clip(img2,0.,1.)
plt.figure()
plt.imshow(img2)