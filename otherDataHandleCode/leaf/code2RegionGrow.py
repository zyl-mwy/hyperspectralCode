import cv2
import matplotlib.pyplot as plt
from  PIL import Image
import numpy as np

def get_x_y(path,n): #path表示图片路径，n表示要获取的坐标个数
    im = Image.open(path)
    plt.imshow(im, cmap = plt.get_cmap("gray"))
    pos=plt.ginput(n)
    return pos   #得到的pos是列表中包含多个坐标元组

#区域生长
def regionGrow(gray, seeds, thresh, p):  #thresh表示与领域的相似距离，小于该距离就合并
    seedMark = np.zeros(gray.shape)
    #八邻域
    if p == 8:
        connection = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    #四邻域
    elif p == 4:
        connection = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    #seeds内无元素时候生长停止
    while len(seeds) != 0:
        #栈顶元素出栈
        pt = seeds.pop(0)
        for i in range(p):
            tmpX = int(pt[0] + connection[i][0])
            tmpY = int(pt[1] + connection[i][1])



            #检测边界点
            if tmpX < 0 or tmpY < 0 or tmpX >= gray.shape[0] or tmpY >= gray.shape[1]:
                continue

            if abs(int(gray[tmpX, tmpY]) - int(gray[pt])) < thresh and seedMark[tmpX, tmpY] == 0:
                seedMark[tmpX, tmpY] = 255
                seeds.append((tmpX, tmpY))
    return seedMark


path = r"2023-10-12_008.png"
img = cv2.imread(path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# hist = cv2.calcHist([gray], [0], None, [256], [0,256])#直方图

# seeds = originalSeed(gray, th=10)
# print(seeds)
seeds=get_x_y(path=path,n=3) #获取初始种子
print("选取的初始点为：")
new_seeds=[]
for seed in seeds:
    print(seed)
    #下面是需要注意的一点
    #第一： 用鼠标选取的坐标为float类型，需要转为int型
    #第二：用鼠标选取的坐标为（W,H），而我们使用函数读取到的图片是（行，列），而这对应到原图是（H,W），所以这里需要调换一下坐标位置，这是很多人容易忽略的一点
    new_seeds.append((int(seed[1]), int(seed[0])))#
    

result= regionGrow(gray, new_seeds, thresh=3, p=8)

#plt.plot(hist)
#plt.xlim([0, 256])
#plt.show()

result=Image.fromarray(result.astype(np.uint8))
result.show()

