import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn import datasets 
from sklearn.model_selection import GridSearchCV
import numpy as np

#导入数据集
dataset = datasets.load_linnerud() 

#数据集读取为dataframe
col_names = dataset['feature_names'] + dataset['target_names']
data = pd.DataFrame(data= np.c_[dataset['data'], dataset['target']], columns=col_names)

#训练集
x_train=np.array(data.loc[:,dataset['feature_names']])
y_train=np.array(data.loc[:,dataset['target_names']])

#回归模型，参数
pls_model_setup = PLSRegression(scale=True)
param_grid = {'n_components': range(1, 4)}

#GridSearchCV 自动调参
gsearch = GridSearchCV(pls_model_setup, param_grid)

#在训练集上训练模型
pls_model = gsearch.fit(x_train, y_train)

#预测
pred = pls_model.predict(x_train)

#打印 coef
print('Partial Least Squares Regression coefficients:',pls_model.best_estimator_.coef_)
