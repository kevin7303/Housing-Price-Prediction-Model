# By: Kevin Wang
# Created: July 1st, 2020
### This is the Data Cleaning
### The Data set was provided by : http://jse.amstat.org/v19n3/decock.pdf
### This dataset was also used in a Kaggle Competition: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview


import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax, skew


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

a = train.columns
print("List in proper method", '[%s]' % ', '.join(map(str, a)))
print(train.columns)
print(train.shape)


#Find Outliers based on EDA results
train = train.drop(train[(train['GrLivArea'] > 4000) &
                         (train['SalePrice'] < 400000)].index)
train = train.drop(train[(train['GrLivArea'] < 3000) &
                         (train['SalePrice'] > 500000)].index)


#Separate feature and target
train_labels = train['SalePrice'].reset_index(drop=True)
train = train.drop(['SalePrice'], axis=1)

df = pd.concat([train, test]).reset_index(drop=True)

#Looking through the Raw Data: Null Values and Data Types
print(df.columns)

#19 columns contain NA values
na_cols = df.loc[:, df.isna().any()]
print(na_cols.shape)
print(na_cols.isna().sum().sort_values(ascending=False))

print(df.count())
print(df.dtypes)

#Data Imputation: Handling NA's

#PoolQC: NA = No pool (Ordinal)
df['PoolQC'] = df['PoolQC'].fillna('None')
print(df.PoolQC.value_counts())
print(df.PoolQC.dtype.name)
d_pool = {'None': 0, 'Fa': 1, 'Gd': 2, 'Ex': 3}
df['PoolQC'] = pd.Categorical(df['PoolQC'], ['Ex', 'Gd', 'Fa', 'None'])
df['PoolQC'].replace(d, inplace=True)
print(df.PoolQC.value_counts())


#MiscFeature: NA = None -> No inherent order have to one hot encode
df['MiscFeature'] = df['MiscFeature'].fillna('None')
print(df.MiscFeature.value_counts())
print(df.MiscFeature.dtype.name)

#Alley: NA = None (Ordinal)
df['Alley'] = df['Alley'].fillna('None')
print(df.Alley.value_counts())
d_alley = {'None': 0, 'Grvl': 1, 'Pave': 2}
df.Alley.replace(d_alley, inplace=True)
df['Alley'] = pd.Categorical(df['Alley'], [0, 1, 2])

#Fence: NA = None (Ordinal)
df['Fence'] = df['Fence'].fillna('None')
print(df.Fence.value_counts())
d_fence = {'None': 0, 'MnWw': 1, 'GdWo': 2, 'MnPrv': 3, 'GdPrv': 4}
df.Fence.replace(d_fence, inplace=True)
df['Fence'] = pd.Categorical(df['Fence'], [0, 1, 2, 3, 4])

#FireplaceQu NA = None (Ordinal)
df['FireplaceQu'].fillna('None', inplace=True)
print(df.FireplaceQu.value_counts())
d_fire = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
df.FireplaceQu.replace(d_fire, inplace=True)
df['FireplaceQu'] = pd.Categorical(df['FireplaceQu'], [0, 1, 2, 3, 4, 5])

#LotFrontage (continuous variable) Analyze best imputation method mean or median
print(df.LotFrontage.isna().sum())
print(df.LotFrontage.mean())
sns.distplot(df.LotFrontage, fit=norm)
(mu, sigma) = norm.fit(df.LotFrontage.dropna())
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(
    mu, sigma)], loc='best')
plt.show()
print(df.LotFrontage.describe())
print(df.LotFrontage.median())

#Remaining NA's
na_cat = ['MSZoning', 'Functional', 'BsmtHalfBath', 'BsmtFullBath',
          'Utilities', 'SaleType', 'KitchenQual', 'Exterior2nd', 'Exterior1st']
na_cont = ['TotalBsmtSF', 'BsmtUnfSF', 'BsmtFinSF2', 'BsmtFinSF1']

for x in na_cat:
    df[x].fillna(df[x].mode()[0], inplace=True)

for x in na_cont:
    df[x].fillna(df[x].mean(), inplace=True)


#Remove outliers that are above 150, mean and median converge at 69
print(df.loc[df['LotFrontage'] < 150, 'LotFrontage'].mean())
print(df.loc[df['LotFrontage'] < 150, 'LotFrontage'].median())

df.LotFrontage.fillna(69, inplace=True)

#Garage - All Related columns since NA = None
garage_col = [column for column in df.columns if 'Garage' in column]
print(garage_col)

for g in garage_col:
    print(g)
    df[g].fillna('None', inplace=True)

for g in garage_col:
    print(g)
    print(df[g].isna().any())

df['GarageArea'] = df['GarageArea'].replace('None', 0)
df['GarageCars'] = df['GarageCars'].replace('None', 0)

#Basement - All related columns to Bsmt since NA = None, First find the 1 discrepency
basement_col = ['BsmtFinType2', 'BsmtExposure',
                'BsmtFinType1', 'BsmtCond', 'BsmtQual']

#Scenario 1 Use the same as BsmtFinType1
print(df.loc[df['BsmtFinType2'].isna(), basement_col])
df.loc[332, 'BsmtFinType2'] = 'GLQ'


#Scenario 2 find the for when BsmtFinType2 is No
print(df.loc[df['BsmtExposure'].isna(), basement_col])
print(df.loc[df['BsmtFinType2'] == 'Unf', 'BsmtExposure'].value_counts())

df.loc[948, 'BsmtExposure'] = 'No'

#Remove remaining NAs
for b in basement_col:
    print(b)
    df[b].fillna('None', inplace=True)

for b in basement_col:
    print(df[b].isna().any())

#Replace NA with Mode

#Electrical - NA != None
print(df.Electrical.describe())
print(df.Electrical.value_counts())
df['Electrical'].fillna(df['Electrical'].mode()[0], inplace=True)
print(df.Electrical.isna().sum())


#Masonry - All related to masonry
mason_col = ['MasVnrArea', 'MasVnrType']
print(df.MasVnrType.value_counts())
print(df.loc[df['MasVnrType'].isna(), mason_col])
print(df.MasVnrArea.describe())

for x in mason_col:
    df[x].fillna(df[x].mode()[0], inplace=True)


#Check for data types
df.dtypes

#Data type Ints that should be Categorical
df['MSSubClass'] = pd.Categorical(df['MSSubClass'])
print(df.MSSubClass)

df['OverallQual'] = pd.Categorical(df['OverallQual'])
df['OverallCond'] = pd.Categorical(df['OverallCond'])

#Data type Object that should be int
df['GarageYrBlt'] = df.GarageYrBlt.replace('None', -1)
df['GarageYrBlt'] = df.GarageYrBlt.astype('int')


#Basement Condition
d_basecond = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4}
print(df.BsmtCond.unique())
df['BsmtCond'].replace(d_basecond, inplace=True)
df['BsmtCond'] = pd.Categorical(df.BsmtCond)
print(df.BsmtCond)

#Basement Exposure
print(df.BsmtExposure.unique())
d_baseexp = {'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}
df['BsmtExposure'].replace(d_baseexp, inplace=True)
df['BsmtExposure'] = pd.Categorical(df.BsmtExposure)
print(df.BsmtExposure)

#Basement Finish Type 1
d_basefin = {'None': 0, 'Unf': 1, 'LwQ': 2,
             'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}
df['BsmtFinType1'].replace(d_basefin, inplace=True)
df['BsmtFinType1'] = pd.Categorical(df.BsmtFinType1)
print(df.BsmtFinType1)

#Basement Finish Type 2
d_basefin = {'None': 0, 'Unf': 1, 'LwQ': 2,
             'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}
df['BsmtFinType2'].replace(d_basefin, inplace=True)
df['BsmtFinType2'] = pd.Categorical(df.BsmtFinType2)
print(df.BsmtFinType2)

#Central Air
d_centralair = {'N': 0, 'Y': 1}
df['CentralAir'].replace(d_centralair, inplace=True)
df['CentralAir'] = pd.Categorical(df.CentralAir)
print(df.CentralAir)

#Basement Quality
d_basequal = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
df['BsmtQual'].replace(d_basequal, inplace=True)
df['BsmtQual'] = pd.Categorical(df.BsmtQual)
print(df.BsmtQual)

#Exterior Condition
d_extcond = {'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4}
df['ExterCond'].replace(d_extcond, inplace=True)
df['ExterCond'] = pd.Categorical(df.ExterCond)
print(df.ExterCond)

#Exterior Quality
d_extcond = {'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4}
df['ExterQual'].replace(d_extcond, inplace=True)
df['ExterQual'] = pd.Categorical(df.ExterQual)
print(df.ExterQual)

#Functional
d_func = {'Sal': 0, 'Sev': 1, 'Maj2': 2, 'Maj1': 3,
          'Mod': 4, 'Min2': 5, 'Min1': 6, 'Typ': 7}
df['Functional'].replace(d_func, inplace=True)
df['Functional'] = pd.Categorical(df.Functional)
print(df.Functional)

#Garage Condition
d_garagecond = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
df['GarageCond'].replace(d_garagecond, inplace=True)
df['GarageCond'] = pd.Categorical(df.GarageCond)
print(df.GarageCond)

#Garage Finish
d_garagefin = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
df['GarageFinish'].replace(d_garagefin, inplace=True)
df['GarageFinish'] = pd.Categorical(df.GarageFinish)
print(df.GarageFinish)

#Garage Quality
d_garagequal = {'None': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}
df['GarageQual'].replace(d_garagequal, inplace=True)
df['GarageQual'] = pd.Categorical(df.GarageQual)
print(df.GarageQual)

#Garage Type
d_garagetype = {'None': 0, 'Detchd': 1, 'CarPort': 2,
                'BuiltIn': 3, 'Basment': 4, 'Attchd': 5,  '2Types': 6}
print(df.GarageType.unique())
df['GarageType'].replace(d_garagetype, inplace=True)
df['GarageType'] = pd.Categorical(df.GarageType)
print(df.GarageType)

#Heating Quality
d_heating = {'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4}
df['HeatingQC'].replace(d_heating, inplace=True)
df['HeatingQC'] = pd.Categorical(df.HeatingQC)
print(df.HeatingQC)

#House Style
d_house = {'1Story': 0, '1.5Unf': 1, '1.5Fin': 2, '2Story': 3,
           '2.5Unf': 4, '2.5Fin': 5, 'SFoyer': 6, 'SLvl': 7}
df['HouseStyle'].replace(d_house, inplace=True)
df['HouseStyle'] = pd.Categorical(df.HouseStyle)
print(df.HouseStyle)

#Kitchen Quality
d_kitchen = {'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4}
df['KitchenQual'].replace(d_kitchen, inplace=True)
df['KitchenQual'] = pd.Categorical(df.KitchenQual)
print(df.KitchenQual)

#LandSlope
d_land = {'Sev': 0, 'Mod': 1, 'Gtl': 2}
df['LandSlope'].replace(d_land, inplace=True)
df['LandSlope'] = pd.Categorical(df.LandSlope)
print(df.LandSlope)


#Lot Configuration
d_lotconf = {'FR3': 0, 'FR2': 1, 'CulDSac': 2, 'Corner': 3, 'Inside': 4}
df['LotConfig'].replace(d_lotconf, inplace=True)
df['LotConfig'] = pd.Categorical(df.LotConfig)
print(df.LotConfig)

#Lot Shape
d_lotshape = {'IR3': 0, 'IR2': 1, 'IR1': 2, 'Reg': 3}
df['LotShape'].replace(d_lotshape, inplace=True)
df['LotShape'] = pd.Categorical(df.LotShape)
print(df.LotConfig)

#Paved Driveway
d_paved = {'N': 0, 'P': 1, 'Y': 2}
df['PavedDrive'].replace(d_paved, inplace=True)
df['PavedDrive'] = pd.Categorical(df.PavedDrive)
print(df.PavedDrive)

#Street
d_street = {'Grvl': 0, 'Pave': 1}
df['Street'].replace(d_street, inplace=True)
df['Street'] = pd.Categorical(df.Street)
print(df.Street)

#Utilities
d_util = {'ELO': 0, 'NoSeWa': 1, 'NoSewr': 2, 'AllPub': 3}
df['Utilities'].replace(d_util, inplace=True)
df['Utilities'] = pd.Categorical(df.Utilities)
print(df.Utilities)

#Differentiating between encoding with label and one hot encoding -> Done in the Model Building script
df_obj = [name for name in df.columns[df.dtypes == 'object']]
print(df_obj)
onehot = ['MSZoning', 'MSSubClass', 'SaleCondition', 'MasVnrType', 'LandContour', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'RoofStyle',
          'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasonVnrType', 'Foundation', 'Heating', 'Electrical', 'MiscFeature', 'SaleType', 'SaleCondtion', 'CentralAir']


def Diff(df_obj, onehot):
    return (list(set(df_obj) - set(onehot)))


encode = Diff(df_obj, onehot)

print(sorted(encode))

#Change features that should be categorical from data type int
print(df.columns[df.dtypes == 'int64'])

cat = ['YearBuilt', 'YearRemodAdd', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr',
       'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars',  'MoSold', 'YrSold', 'MiscVal']

for x in cat:
    df[x] = pd.Categorical(df[x])

#New Features combination
df['Total_Square_Feet'] = (df['1stFlrSF'] + df['2ndFlrSF'] + df['TotalBsmtSF'])
df['Total_Bathrooms'] = (df['FullBath'] + (0.5 * df['HalfBath']) +
                         df['BsmtFullBath'] + (0.5 * df['BsmtHalfBath']))

#Check df attributes
print(df.dtypes)
print(df.shape)
print(df.isna().sum())

#Multicollinearity
print(df.dtypes)
df = df.drop(['GarageYrBlt'], axis = 1)

#Split train and test again
train_clean = df.iloc[:len(train_labels), :]
train_clean['SalePrice'] = train_labels.values
test_clean = df.iloc[len(train_labels):, :]

#Save new CSV
train_clean.to_csv('train_model.csv', index=False)
test_clean.to_csv('test_model.csv', index=False)
