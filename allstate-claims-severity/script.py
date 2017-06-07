import os, gc
import numpy as np
import pandas as pd
from scipy.stats import boxcox

from sklearn import datasets, metrics, model_selection
from sklearn.decomposition import NMF
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder

from pylightgbm.models import GBMRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

df_train = pd.read_csv("data/train.csv")
print('Train data shape', df_train.shape)

df_test = pd.read_csv("data/test.csv")
print('Test data shape', df_test.shape)

y = np.log(df_train['loss']+1).as_matrix().astype(np.float)
id_test = np.array(df_test['id'])

df = df_train.append(df_test, ignore_index=True)
del df_test, df_train
gc.collect()

print('Merged data shape', df.shape)
dropped_columns = ['loss', 'id',]
df.drop(labels=dropped_columns, axis=1, inplace=True)
le = LabelEncoder()


cols_d = []
n_cols_d = []

for col in df.columns:
    if str(df[col].dtype) == 'object':
        df[col] = LabelEncoder().fit_transform(df[col])
        cols_d.append(col)
    else:
        n_cols_d.append(col)


print('train-test split')
df_train, df_test = df.iloc[:len(y)], df.iloc[len(y):]
del df
gc.collect()

print('train-validation split\n')
X = df_train.as_matrix()
X_train, X_valid, y_train, y_valid = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)
X_test = df_test.as_matrix()


print('Train shape', X_train.shape)
scaler = StandardScaler()
print('Validation shape', X_valid.shape)
print('Test shape', X_test.shape)

from xgboost import XGBRegressor
lr = XGBRegressor(max_depth=10, objective='reg:linear', min_child_weight=1,learning_rate=0.075,colsample_bytree=0.7)

# lr.fit(X_train, y_train)

# print("Mean absolute Error: ", metrics.mean_absolute_error(y_true=(np.exp(y_valid)-1),
#                                                            y_pred=(np.exp(lr.predict(X_valid))-1)))

# import operator
# features = lr.booster()
# fscores = features.get_fscore()
# importance_features = sorted(fscores.items(), key=operator.itemgetter(1))
#
# print importance_features

import numpy as np

# columns = [  75, 23, 84, 29, 67, 59, 45, 104, 82, 58, 68, 70, 106, 44, 76, 31, 43, 81, 42, 22, 73, 57, 53, 83, 74, 80,
#              101, 63, 71, 78, 26, 91, 30, 24, 25, 85, 100, 40, 27, 33, 39, 56, 92, 38, 103, 52, 88, 37, 54, 114, 51, ]
columns = [75, 23, 84, 29, 67, 59, 45, 104, 82, 58, 68, 70, 106, 44, 76,]

# columns = []
X_train_delete = np.delete(X_train, columns, 1)
X_test_delete = np.delete(X_test, columns, 1)
X_valid_delete = np.delete(X_valid, columns, 1)

nmf = NMF(n_components=50)
matrix = nmf.fit_transform(X_train_delete)
print("after nmf")

from sklearn.preprocessing import normalize
l2 = normalize(matrix)

lr.fit(matrix, y_train)
print("Mean absolute Error after feature delete: ", metrics.mean_absolute_error(y_true=(np.exp(y_valid)-1),
                                                           y_pred=(np.exp(lr.predict(normalize(nmf.transform(X_valid_delete))))-1)))



y_test_preds = lr.predict(X_test_delete)
y_test_preds=(np.exp(y_test_preds)-1)


df_submision = pd.read_csv('data/sample_submission.csv')
df_submision['loss'] = y_test_preds

df_submision.to_csv('submission.csv',index=False)

#3 XGBRegressor, depth10 = ('Mean absolute Error: ', 1144.0308346038405), Results = 1139.15931
#5 XGBRegressor, depth10, remove columns < 10 ('Mean absolute Error: ', 1144.4369070509115)
#5 XGBRegressor, depth10, remove columns < 100 ('Mean absolute Error: ', 1159.9935812814172)
#4 XGBRegressor, depth20 = ('Mean absolute Error: ', 1191.9035212927911), Results = ?

#1 Linear Regression = ('Mean absolute Error: ', 1287.6157692360255)
#2 SVR = ('Mean absolute Error: ', 8030.2037953985991)
#2 XGBRegressor = ('Mean absolute Error: ', 1196.9398794925139), Results = 1187.94083
#4 XGBRegressor, max_depth=10, min_child_weight=40 = ('Mean absolute Error: ', 1141.818039921912), Results = ?
#5 XGBRegressor, NMF=20 ('Mean absolute Error: ', 1285.3242496578634)
#5 XGBRegressor, NMF=50 ('Mean absolute Error: ', 1233.440964167813)
#5 XGBRegressor, NMF=100 ('Mean absolute Error: ', 1217.0514165006134)
#5 XGBRegressor, NMF=100, alpha=0.0002, beta=0.02 ('Mean absolute Error: ', 1217.0514165006134)
