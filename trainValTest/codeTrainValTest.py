import pandas as pd
import numpy as np

excel = pd.read_excel(r'E:\my_project\hyperspectralData\medician\leaf\Final.xlsx')

data = excel.values[:, 1:]
# print(data.shape[0])
dataRandom = np.random.permutation(data.shape[0])
# print(len(dataRandom))
length = dataRandom.shape[0]
dataTrain = data[dataRandom[:8*(length//10)]]
dataVal = data[dataRandom[8*(length//10):9*(length//10)]]
dataTest = data[dataRandom[9*(length//10):]]
print(length, dataTrain.shape[0], dataVal.shape[0], dataTest.shape[0], dataTrain.shape[0]+dataVal.shape[0]+dataTest.shape[0])

data_df = pd.DataFrame(dataTrain)
data_df.columns = excel.columns[1:]
data_df.to_excel(r'E:\my_project\hyperspectralData\medician\leaf\FinalTrain.xlsx')

data_df = pd.DataFrame(dataVal)
data_df.columns = excel.columns[1:]
data_df.to_excel(r'E:\my_project\hyperspectralData\medician\leaf\FinalVal.xlsx')

data_df = pd.DataFrame(dataTest)
data_df.columns = excel.columns[1:]
data_df.to_excel(r'E:\my_project\hyperspectralData\medician\leaf\FinalTest.xlsx')