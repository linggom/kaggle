
# coding: utf-8

# Righnow candidate is : SVR with PCA n_components = 10 with results 0.23706564506631833

# In[ ]:
print("starting import ")
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression, TheilSenRegressor,RANSACRegressor,HuberRegressor
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import GradientBoostingRegressor


# In[ ]:

print("starting load")

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')


# In[ ]:

test.head(5)


# In[ ]:

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import OneClassSVM


# In[ ]:

train_labels = train.iloc[:,1]
train_feats = train.iloc[:, 2:]

test_labels = test.iloc[:,1]
test_feats = test.iloc[:, 1:]


# In[ ]:
print("starting feature extraction ")

train_feats_encode = []
train_feats_encode.append(train.iloc[:, 0])
for i in range(train_feats.shape[1]):
    arr = train_feats.iloc[:, i]
    if arr.dtype == 'O':
        lblencod = LabelEncoder()
        arr = lblencod.fit_transform(arr)
    train_feats_encode.append(arr)
train_feats_encode = np.array(train_feats_encode).T

test_feats_encode = []
test_feats_encode.append(test.iloc[:, 0])
for i in range(test_feats.shape[1]):
    arr = test_feats.iloc[:, i]
    if arr.dtype == 'O':
        lblencod = LabelEncoder()
        arr = lblencod.fit_transform(arr)
    test_feats_encode.append(arr)
test_feats_encode = np.array(test_feats_encode).T


# In[ ]:

cellen=['ID', 'X47','X95','X314','X315','X232','X119','X311','X76','X329','X238','X340','X362','X137']

# cellen = ["X28", "X29", "X54", "X76", "X118", "X119", "X127", "X136", "X162", "X166", "X178", "X232", "X234", "X250", "X261", "X263", "X272", "X275", "X276", "X279", "X313", "X314", "X316", "X328", "X378",]


# In[ ]:

# train_feats_encode = train[cellen].as_matrix()
# test_feats_encode = test[cellen].as_matrix()


# In[ ]:

from sklearn.ensemble import IsolationForest

rng = np.random.RandomState(42)

print("starting remove outliers ")

# Example settings
n_samples = 200
outliers_fraction = 0.25
clusters_separation = [0, 1, 2]
outl = IsolationForest(max_samples=n_samples,
                                        contamination=outliers_fraction,
                                        random_state=rng)
outl.fit(train_feats_encode)


# In[ ]:

qw = outl.predict(train_feats_encode)
np.unique(qw)


# In[ ]:

train_feats_encode = train_feats_encode[qw == 1]
train_labels = train_labels[qw == 1]


# In[ ]:

train_X, eval_X, train_y, eval_y = train_test_split(train_feats_encode, train_labels)


# In[149]:
print("starting modelling ")

def check(model):
    pred_train = model.predict(train_X)
    score_train = r2_score(train_y, pred_train)
    pred_eval = model.predict(eval_X)
    score_eval = r2_score(eval_y, pred_eval)
    print(model.__class__.__name__)
    print("r2 train = ", score_train)
    print("r2 eval = ", score_eval)
    print("====================================================\n")
    return score_train, score_eval, pred_train, pred_eval


# In[150]:

def eval_model():
    print("TheilSenRegressor ")

    model_theilsen = TheilSenRegressor(random_state=42)
    model_theilsen.fit(train_X, train_y)
    train_theilsen, test_theilsen, pred_train_theilsen, pred_eval_theilsen = check(model_theilsen)

    print("RANSACRegressor ")
    model_ransac = RANSACRegressor(random_state=42)
    model_ransac.fit(train_X, train_y)
    train_ransac, test_ransac, pred_train_ransac, pred_eval_ransac = check(model_ransac)

    print("HuberRegressor ")
    model_huber = HuberRegressor()
    model_huber.fit(train_X, train_y)
    train_huber, test_huber, pred_train_huber, pred_eval_huber = check(model_huber)

    print("LinearRegression ")
    model_linear = LinearRegression()
    model_linear.fit(train_X, train_y)
    train_lr, test_lr, pred_train_lr, pred_eval_lr = check(model_linear)

    print("SGDRegressor ")
    model_sgd = SGDRegressor()
    model_sgd.fit(train_X, train_y)
    train_sgd, test_sgd, pred_train_sgd, pred_eval_sgd = check(model_sgd)

    model_xgb = XGBRegressor(seed = 0,
      colsample_bytree = 0.7,
      subsample = 0.9,
      eta = 0.005,
      max_depth = 4,
      num_parallel_tree = 1,
      min_child_weight = 1, objective='reg:linear', base_score=np.mean(train_labels))
    model_xgb.fit(train_X, train_y)
    train_xgb, test_xgb, pred_train_xgb, pred_eval_xgb = check(model_xgb)

    model_gbr = GradientBoostingRegressor()
    model_gbr.fit(train_X, train_y)
    train_gbr, test_gbr, pred_train_gbr, pred_eval_gbr = check(model_gbr)



    columns=['lr', 'sgd', 'xgb', 'gbr',  'thelisen', 'ransac', 'huber']

    test_res = (pred_eval_lr, pred_eval_sgd, pred_eval_xgb, pred_eval_gbr, pred_eval_theilsen, pred_eval_ransac, pred_eval_huber)

    train_res = (pred_train_lr, pred_train_sgd, pred_train_xgb, pred_train_gbr, pred_train_theilsen, pred_train_ransac, pred_train_huber)

    train_res_pd = pd.DataFrame(data=np.column_stack(train_res),
                  columns=columns)

    test_res_pd = pd.DataFrame(data=np.column_stack(test_res),
                  columns=columns)
    models = [model_linear, model_sgd, model_xgb, model_gbr, model_theilsen, model_ransac, model_huber]
    return train_res_pd, test_res_pd, models


# In[151]:

train_pd, eval_pd, models = eval_model()


# In[96]:

def frm(dt):
    x = 0.1 * dt['lr']  + 0.75 * dt['xgb'] + 0.15 * dt['gbr']
    return x


# In[97]:

print(r2_score(train_y, train_pd.median(axis=1)))
print(r2_score(eval_y, eval_pd.median(axis=1)))
print(r2_score(train_y, frm(train_pd)))
print(r2_score(eval_y, frm(eval_pd)))


# In[98]:

all_models = []
for model in models:
    nme = str(model.__class__.__name__)
    if nme != 'KerasRegressor':
        model.fit(train_feats_encode, train_labels)
        print(nme)
        print(r2_score(train_labels, model.predict(train_feats_encode)))
        all_models.append(model)


# In[36]:

preds = []
for model in models:
    preds.append(model.predict(train_feats_encode))
columns=['lr', 'sgd', 'xgb', 'gbr',  'thelisen', 'ransac', 'huber']
dt = pd.DataFrame(data=np.array(preds).T, columns=columns)
r2_score(train_labels, frm(dt))


# In[22]:

preds = []
columns=['lr', 'sgd', 'xgb', 'gbr', 'thelisen', 'ransac', 'huber']

for model in models:
    preds.append(model.predict(test_feats_encode))

dt = pd.DataFrame(data=np.array(preds).T, columns=columns)


# In[23]:

res = dt[['lr', 'xgb', 'gbr',  'ransac']]


# In[24]:

output = pd.DataFrame({'id': test['ID'].astype(np.int32)})


# In[29]:

output['y'] = frm(res)


# In[30]:

output.to_csv('results/xgboost/ensemble[%s][%s].csv' % ('ensemble','qwe12'), index=False)
