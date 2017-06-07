import os, gc
import numpy as np
import pandas as pd
from scipy.stats import boxcox

from sklearn import datasets, metrics, model_selection
from sklearn.decomposition import NMF
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder

from pylightgbm.models import GBMRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

df_train = pd.read_csv("data/train.csv")
print('Train data shape', df_train.shape)

df_test = pd.read_csv("data/test.csv")
print('Test data shape', df_test.shape)

y = np.log(df_train['loss'] + 1).as_matrix().astype(np.float)
id_test = np.array(df_test['id'])

df = df_train.append(df_test, ignore_index=True)
del df_test, df_train
gc.collect()

print('Merged data shape', df.shape)
df.drop(labels=['loss', 'id'], axis=1, inplace=True)
le = LabelEncoder()

for col in df.columns.tolist():
    if 'cat' in col:
        df[col] = le.fit_transform(df[col])

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

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(X_train.shape[1], input_dim=X_train.shape[1], init='normal', activation='relu'))
    model.add(Dense(1, init='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
# evaluate model with standardized dataset

estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=100, batch_size=5, verbose=1)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(estimator, X_train, y_train, cv=kfold)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
import ipdb;ipdb.set_trace()

#
# y_test_preds = lr.predict(X_test)
# y_test_preds = (np.exp(y_test_preds) - 1)
#
# df_submision = pd.read_csv('data/sample_submission.csv')
# df_submision['loss'] = y_test_preds
#
# df_submision.to_csv('submission.csv', index=False)

# 1 Linear Regression = ('Mean absolute Error: ', 1287.6157692360255)
# 2 SVR = ('Mean absolute Error: ', 8030.2037953985991)
# 2 XGBRegressor = ('Mean absolute Error: ', 1196.9398794925139), Results = 1187.94083
# 3 XGBRegressor, depth10 = ('Mean absolute Error: ', 1144.0308346038405), Results = 1139.15931
# 4 XGBRegressor, depth20 = ('Mean absolute Error: ', 1191.9035212927911), Results = ?
# 4 XGBRegressor, max_depth=10, min_child_weight=40 = ('Mean absolute Error: ', 1141.818039921912), Results = ?
# 5 XGBRegressor, NMF=20 ('Mean absolute Error: ', 1285.3242496578634)
# 5 XGBRegressor, NMF=50 ('Mean absolute Error: ', 1233.440964167813)
# 5 XGBRegressor, NMF=100 ('Mean absolute Error: ', 1217.0514165006134)
# 5 XGBRegressor, NMF=100, alpha=0.0002, beta=0.02 ('Mean absolute Error: ', 1217.0514165006134)
