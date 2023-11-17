import pandas as pd
import numpy as np
import os
 
# 读取Excel数据
df = pd.read_excel('test.xlsx')
# print(df.columns)
 
# 将DataFrame对象转换为numpy数组
numpy_array = df.values
 
# 转换为二维数组
two_dimensional_array = np.array(numpy_array)
my_array = np.array(two_dimensional_array[:, :9])
# print(my_array)
dirListLeaf = os.listdir(r'E:\my_project\hyperspectralData\medician\leaf\drawRigion\runs\labelme2coco\hyperspectralDataSplit')
# dirListSoil = os.listdir(r'E:\my_project\hyperspectralData\medician\soil\drawRigion\runs\labelme2coco\hyperspectralDataSplit')
for i in range(my_array.shape[0]):
    my_array[i, 1] = dirListLeaf[2*i][:-4]
# print(my_array)

data_df = pd.DataFrame(my_array)
data_df.columns = df.columns[:9]
# df.to_excel('tb.xlsx',            # 路径和文件名
#             sheet_name='tb1',     # sheet 的名字
#             float_format='%.2f',  # 保留两位小数
#             na_rep='我是空值')     # 空值的显示
data_df.to_excel('test2.xlsx')     # 空值的显示
