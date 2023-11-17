from torch.utils.data import DataLoader

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import matplotlib.pyplot as plt
import numpy as np
import tqdm

plt.rcParams['font.sans-serif'] = ['SimHei']

def test(dataset, model, device, criterion, criterion1, name):
    model = model.to(device)
    model.eval()
    test_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    outputList = []
    labelList = []
    
    # 模型测试
    lossTatol = 0
    t = tqdm.tqdm(enumerate(test_loader),desc = f'[Test]') # ,loss:{loss.item()}
    for step, (img, label) in t:
        output = model(img.permute(0, 1, 4, 2, 3).to(device))
        # loss = criterion(output, label.to(device))
        loss = criterion(output, label.to(device)) + criterion1(output, label.to(device))
        
        lossTatol += loss.item()
        outputList.append(np.array(output[0].cpu().detach().numpy()))
        labelList.append(np.array(label[0].cpu().detach().numpy()))

    lossAverage = lossTatol/(step+1)
    outputList = np.array(outputList)
    labelList = np.array(labelList)
    # https://www.zhihu.com/question/330027160
    '''
    均方根误差 (RMSE)：一般而言，取值越低越好，一般取值小于0.5为表现良好；
    均方误差 (MSE)：也是一般而言取值越低越好，一般取值小于0.2为表现良好；
    平均绝对误差（MAE）：一般取值小于0.1为表现良好；
    平均相对百分误差（MAPE）：一般取值小于10%为表现良好；
    决定系数R2：一般取值大于0.8为表现良好。
    '''
    # print(outputList)
    MSE1 = mean_squared_error(outputList[:, 0], labelList[:, 0])
    MAE1 = mean_absolute_error(outputList[:, 0], labelList[:, 0])
    R21 = r2_score(outputList[:, 0], labelList[:, 0])    
    MSE2 = mean_squared_error(outputList[:, 1], labelList[:, 1])
    MAE2 = mean_absolute_error(outputList[:, 1], labelList[:, 1])
    R22 = r2_score(outputList[:, 1], labelList[:, 1])  
    print('lossAverage:', lossAverage, 'loss:', loss.item(), 'MSE1:', MSE1, 'MAE1:', MAE1, 'R21:', R21, 'MSE2:', MSE2, 'MAE2:', MAE2, 'R22:', R22)
    # print(outputList)
    # print(labelList)
    # plt.plot(outputList)
    # plt.plot(labelList)
    plt.figure(figsize=(6, 3))
    plt.subplot(121)
    plt.title(name+'叶绿素')
    plt.plot([i/100 for i in range(100)], [i/100 for i in range(100)])
    plt.plot([i/100 for i in range(100)], [i/100+0.05 for i in range(100)])
    plt.plot([i/100 for i in range(100)], [i/100-0.05 for i in range(100)])
    plt.scatter(outputList[:, 0], labelList[:, 0])
    plt.subplot(122)
    plt.title(name+'氮素')
    plt.plot([i/100 for i in range(100)], [i/100 for i in range(100)])
    plt.plot([i/100 for i in range(100)], [i/100+0.05 for i in range(100)])
    plt.plot([i/100 for i in range(100)], [i/100-0.05 for i in range(100)])
    plt.scatter(outputList[:, 1], labelList[:, 1])
    plt.show()