# By: Kevin Wang
# Created: July 1st, 2020
### This is the Data Cleaning
### The Data set was provided by : http://jse.amstat.org/v19n3/decock.pdf
### This dataset was also used in a Kaggle Competition: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview
###ML models attempted were Decision Tree Classifier, Gradient Boosting Classifier, KNN and RandomForest

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, cross_val_predict
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor, StackingRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.stats import skew

np.random.seed(42)

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

#Import cleaned data
train = pd.read_csv('train_model.csv')
print(train.shape)
test = pd.read_csv('test_model.csv')
print(test.shape)


train_labels = train['SalePrice'].reset_index(drop=True)
train = train.drop(['SalePrice'], axis=1)

df = pd.concat([train, test]).reset_index(drop=True)

#Check data types
print(df.dtypes)

#One hot encode due to Sklearn categorical variable limitation (Sklearn Decision trees treat categorical variable as continuous)
onehot = [x for x in df.columns[df.dtypes == 'object']]
print(onehot)

df = pd.get_dummies(df, dtype = 'int64', columns=['MSZoning', 'LandContour', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'RoofStyle', 'RoofMatl', 'Exterior1st',
                                 'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating', 'Electrical', 'GarageFinish', 'GarageQual', 'PoolQC', 'MiscFeature', 'SaleType', 'SaleCondition'])

#Save and drop ID column
id_col = df['Id']
df = df.drop(['Id'], axis=1)

test = df.iloc[len(train_labels):, :]
df = df.iloc[:len(train_labels), :]
df['SalePrice'] = train_labels.values

#Check number of columns
print(df.shape)
print(test.shape)

#StandardScaling to scale the features
feat = df.drop(['SalePrice'], axis=1).values
rb_scale = RobustScaler()
X = rb_scale.fit_transform(feat)
sale_log = np.log1p(df['SalePrice'])
y = sale_log.values


#Test Set
#Save and drop ID column
test_scale = rb_scale.transform(test)
print(test_scale)


#Root Mean Squared Log Error Helper Function
def cv_rmse(model, x):
    rmse = (np.mean(cross_val_score(model, X, y,
                                    scoring='neg_root_mean_squared_error', cv=x)) * -1)
    return rmse


#Ridge Regression
rr = Ridge()
score = cv_rmse(rr, 5)
print(score)  

#Lasso Regression
lr = Lasso()
score = cv_rmse(lr, 5)
print(score)  

#Elastic Net
el = ElasticNet()
score = cv_rmse(el, 5)
print(score)

#Gradient Boosting Regressor
gbm = GradientBoostingRegressor()
score = cv_rmse(gbm, 5)
print(score)  

#Random Forest Regressor
rf = RandomForestRegressor()
score = cv_rmse(rf, 5)
print(score)  


#Tuning - GridSearchCV

#Ridge Regression
alpha = []
for i in range(1, 10000):
    alpha.append(i/100)
param_grid = {'alpha': alpha}

grid_search = GridSearchCV(rr, param_grid=param_grid,
                           scoring='neg_root_mean_squared_error', cv=5)
grid_search.fit(X, y)
print("Best score is {}".format(grid_search.best_params_))
print("Best score is {}".format(grid_search.best_score_))  

#Lasso Regression
alpha = []
for i in range(1, 1000):
    alpha.append(i/10000)
param_grid = {'alpha': alpha}

grid_search = GridSearchCV(lr, param_grid=param_grid,
                           scoring='neg_root_mean_squared_error', cv=5)
grid_search.fit(X, y)
print("Best score is {}".format(grid_search.best_params_)) 
print("Best score is {}".format(grid_search.best_score_)) 

#Elastic Net
alpha = []
for i in range(1, 1000):
    alpha.append(i/10000)
param_grid = {'alpha': alpha}

grid_search = GridSearchCV(el, param_grid=param_grid,
                           scoring='neg_root_mean_squared_error', cv=5)
grid_search.fit(X, y)
print("Best score is {}".format(grid_search.best_params_))
print("Best score is {}".format(grid_search.best_score_))


#Gradient Boosting Regressor
param_grid = {
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "max_depth": [1, 5, 10, 20, 40],
    "n_estimators": [2000],
    "verbose": [1]
}

grid_search = GridSearchCV(gbm, param_grid=param_grid,
                           scoring='neg_root_mean_squared_error', cv=5)
# {'learning_rate': 0.15, 'max_depth': 3, 'verbose': 1, 'n_estimator' = 300}
print("Best score is {}".format(grid_search.best_params_))
print("Best score is {}".format(grid_search.best_score_))

#Random Forest Regressor
param_grid={
        'max_depth':[5, 10, 20 ,40, 80],
        'max_features':['sqrt','log2'], 
        'n_estimators':[200], 
        'min_samples_leaf':range(5,15,5), 
        'min_samples_split':range(10,30,10), 
        'criterion':['mse', 'mae'] 
    }

grid_search = GridSearchCV(rf, param_grid=param_grid,
                           scoring='neg_root_mean_squared_error', cv=5)
print("Best score is {}".format(grid_search.best_params_))
print("Best score is {}".format(grid_search.best_score_))


#Tuned Models
#Ridge Regression
rr = Ridge(alpha=12.63)
score_rr = cv_rmse(rr, 5)
print(score_rr)
rr.fit(X, y)

#Lasso Regression
lr = Lasso(alpha=0.0005)
score_lr = cv_rmse(lr, 5)
print(score_lr)
lr.fit(X, y)

#Elastic Net
el = ElasticNet(alpha=0.0008)
score_el = cv_rmse(el, 5)
print(score_el)
el.fit(X, y)

#Gradient Boosting Regressor
gbm = GradientBoostingRegressor(n_estimators=2000, learning_rate=0.03, max_depth=4,
                                max_features='sqrt', min_samples_leaf=15, min_samples_split=14, loss='huber')
score_gbm = cv_rmse(gbm, 5)
print(score_gbm)
gbm.fit(X, y)

#Random Forest Regressor
rf = RandomForestRegressor(n_estimators=200, max_depth= 80, max_features='sqrt', min_samples_leaf = 5, min_samples_split = 10)
score_rf = cv_rmse(rf, 5)
print(score_rf)
rf.fit(X, y)

#Ensemble Method 
#Stacking Regressor
class_stack = [('Gradient Boosting Regressor',gbm), ('Ridge', rr), ('Lasso', lr), ('Elastic Net', el)]
# stack_gen = StackingRegressor(estimators=class_stack)
stack_gen = StackingRegressor(estimators=class_stack)
score_sg = cv_rmse(stack_gen, 5)
print(score_sg)
stack_gen.fit(X,y)


#Voting Regressor
classifiers = [('Lasso', lr),('Gradient Boosting Regressor', gbm), ('Ridge Regression', rr), ('ElasticNet', el), ('Stacking Regressor', stack_gen)]
vr = VotingRegressor(estimators=classifiers, weights = [.1,.1,.1,.1,.6])
score_vr = cv_rmse(vr, 5)
print('Voting Regressor', score_vr)

#Fit and predict
vr.fit(X, y)
pred = vr.predict(test_scale)
print(pred)

kaggle = np.expm1(pred)
print(kaggle)

# #Submission for Kaggle
# submission = pd.read_csv("sample_submission.csv")
# submission.shape

# submission.iloc[:, 1] = kaggle
# print(submission.shape)

# submission.to_csv("submission.csv", index=False)

