# Housing-Price-Prediction-Model
Prediction Model Built with Python using the Ames Housing Dataset
Project focused on: Data cleaning, Exploratory Data Analysis, Feature Engineering and Machine Learning Model Building

# Project Overview 
* Cleaned and analyzed data provided by Kaggle's Housing competition: http://jse.amstat.org/v19n3/decock.pdf
* Provided detailed visual analysis of the Ames Housing dataset to reach relational insight between features and data structures
* Engineered features such as Total Square footage and total bathroom numbers by combining columns into insightful information 
* Optimized Random Forest, Gradient Boosting Regressor, Ridge Regression, Lasso Regression and Elastic Net using GridsearchCV to reach the best model
 

## Code and Resources Used 
**Python Version:** 3.7  
**Packages:** pandas, numpy, sklearn, scipy, matplotlib, seaborn  
**Original Kaggle Dataset:** https://www.kaggle.com/c/house-prices-advanced-regression-techniques

## Data Overview
Dataset is divided into Train and Test sets to facilitate the Kaggle competition:
Train Set includes 1460 rows and 81 columns (including the target variable SalePrice)
Test Set includes 1459 rows and 80 columns (Excluding the target variable SalePrice)

Target Variable: 
SalePrice - the property's sale price in dollars. This is the target variable that you're trying to predict.

Columns (features) are:
Id, MSSubClass, MSZoning, LotFrontage, LotArea, Street, Alley, LotShape, LandContour, Utilities, LotConfig, LandSlope, Neighborhood, Condition1, Condition2, BldgType, HouseStyle, OverallQual, OverallCond, YearBuilt, YearRemodAdd, RoofStyle, RoofMatl, Exterior1st, Exterior2nd, MasVnrType, MasVnrArea, ExterQual, ExterCond, Foundation, BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinSF1, BsmtFinType2, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, Heating, HeatingQC, CentralAir, Electrical, 1stFlrSF, 2ndFlrSF, LowQualFinSF, GrLivArea, BsmtFullBath, BsmtHalfBath, FullBath, HalfBath, BedroomAbvGr, KitchenAbvGr, KitchenQual, TotRmsAbvGrd, Functional, Fireplaces, FireplaceQu, GarageType, GarageYrBlt, GarageFinish, GarageCars, GarageArea, GarageQual, GarageCond, PavedDrive, WoodDeckSF, OpenPorchSF, EnclosedPorch, 3SsnPorch, ScreenPorch, PoolArea, PoolQC, Fence, MiscFeature, MiscVal, MoSold, YrSold, SaleType, SaleCondition

## Data Cleaning
I did extensive data cleaning in order to facilitate the exploratory analysis and the model building process:

*	Coerced the features to their respective correct data types after handling unique complications
* Encoded categorical features with appropriate ordinal values when ordinality is present
* Encoded categorical values with one hot encoding when ordinality is not present
*	Performed data imputation based on aggregate functions such as mean and median based on their deviation to a normal distribution
* Used data visualization to safely remove outliers that were most likely erroneous 
*	Dropped irrelevant features such as ID
* Dropped highly correlated features to combat multicollinearity
*	Performed transformation to target variable to remove skewness 
*	Improved model efficiency by using Standard Scaling to better handle outliers

## EDA
The expoloraty data analysis was done to better visualize and understand data set before undergoing the model building process.

Explored the target data and its distribution. Created heat map correlation matrix to better understand the data and find potential cases of multicollinearity. Created plots to visually display feature relations and possible correlations:
Below are some of the graphs created with seaborn:


![alt text](https://github.com/kevin7303/Housing-Price-Prediction-Model/blob/master/CorrMatrix.PNG "Correlation Matrix")
![alt text](https://github.com/kevin7303/Housing-Price-Prediction-Model/blob/master/SalePrice_Distribution.PNG "Sale Price Distribution")
![alt text](https://github.com/kevin7303/Housing-Price-Prediction-Model/blob/master/SalePrice_vs_LivingArea.PNG "Sale Price vs Living Area")



## Model Building 
I wanted to create a model that would accurate predictions for housing prices and that would score well on based on the Kaggle competition metrics.

*Evaluation Metric*

The specific metric used to evaluate the model was Root Mean Squared Error on the Log value of the SalePrice.
This metric was chosen to standardize the effect of error regardless of expensive or cheap house.

**Steps Taken**

Performed One hot encoding on the categorical variables in order to accomodate Sklearn Decision trees treatment of categorical variables as continuous

Used an assortment of simple and complex regression models to create a robust model to accurately predict SalePrice.
Started with base model to evaluate the performance of an unoptimzed and unfitted model on the problem and later used GridSearchCV to optimize the hyper parameters.
All of these were done using a 5 fold cross validation.

Models Used:
* **Ridge Regression**
* **Lasso Regression**
* **Elastic Net Regression**
* **Gradient Boosting Regression**
* **Random Forest Regression**

Ensemble Methods Used:
* **Stacking Regressor**
* **Voting Regressor**


## Model performance
Tuning was done on all of the functions above to increase prediction accuracy.


**Model Parameters

*	**Ridge Regression** – Alpha: 12.63 
*	**Lasso Regression** – Alpha: 0.0005 
*	**ElasticNet Regression** – Alpha: 0.0008
*	**Gradient Boosting Regression** – n_estimators=2000, learning_rate=0.03, max_depth=4,max_features='sqrt', min_samples_leaf=15, min_samples_split=14, loss='huber'
* **Random Forest Regression** - n_estimators=200, max_depth= 80, max_features='sqrt', min_samples_leaf = 5, min_samples_split = 10

**Ensemble Models**
A Stacking Regressor was used to aggregate the best aspects of each of the models and create a blended model that resulted in higher accuracy.
After many iterations, and model combinations, I decided to drop the Random Forest algorithm from the Stacking Regressor due to better accuracy.

The Stacking Regressor incldues: Gradient Boosting Regression, Ridge Regression, Lasso Regression and Elastic Net Regression.

A Voting Regressor was then used to further increase accuracy by distributing the prediction weight between the following models:
* Lasso = 10%
* Gradient Boosting Regression = 10%
* Ridge Regression = 10%
* Elastic Net = 10%
* Stacking Regression (Blended model) = 60%


**The result led to an RMSE of 0.12590 calculated by Kaggle. On July 3rd, this was within the Top 10% of best scoring models**


