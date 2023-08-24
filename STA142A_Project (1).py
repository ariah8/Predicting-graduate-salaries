#!/usr/bin/env python
# coding: utf-8

# In[488]:


import pandas as pd

salaryEngineering = pd.read_csv("Engineering_graduate_salary.csv")
salaryEngineering


# In[489]:


from sklearn import preprocessing
# colnames of X
colnames = salaryEngineering.columns.values[:-1]

# convert categorical values to numeric labels
for i in colnames:
    if(salaryEngineering[i].dtypes in ['string','object']):
        label_encoder = preprocessing.LabelEncoder()
        label_encoder.fit(salaryEngineering[i])
        salaryEngineering[i] = label_encoder.transform(salaryEngineering[i])
salaryEngineering


# In[556]:


from sklearn.feature_selection import RFE
from sklearn.svm import SVR
import warnings
warnings.filterwarnings('ignore')

# use rfe function to get rankings to select features
X = salaryEngineering.iloc[:, :-1].select_dtypes(include = ['number'])
y = salaryEngineering.iloc[:, -1:]
estimator = SVR(kernel="linear")
selector = RFE(estimator, n_features_to_select=5, step=1)
selector = selector.fit(X, y)
rankings = pd.Series(selector.ranking_, index = X.columns)
best = rankings[rankings > 10]
best


# In[490]:


# drop columns that is not significant
columns_to_drop = best.index.values.tolist()
salaryEngineering.drop(columns=columns_to_drop, inplace=True)
salaryEngineering


# In[507]:


import seaborn as sns
plt.figure(figsize=(25,20))
sns.heatmap(salaryEngineering.corr(), annot=True, cmap='viridis')
plt.savefig("heatmap")


# In[492]:


corr_mat = salaryEngineering.corr()
corr_mat["Salary"]


# In[555]:


from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

# feature selection
X = salaryEngineering.iloc[:, :-1].select_dtypes(include = ['number'])
y = salaryEngineering.iloc[:, -1:]

standardScaler = StandardScaler()
standardScaler.fit(X)
X_trans = standardScaler.transform(X)
X = pd.DataFrame(X_trans, columns=X.columns)

model = ExtraTreesClassifier()
model.fit(X,np.ravel(y,order='C'))

#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
print(feat_importances.nlargest(10))
plt.rcParams["figure.figsize"] = (15,10)
plt.savefig("extraTrees")
plt.show()
X


# In[495]:


## countplot of salary
sns.countplot(x='Salary', data=salaryEngineering, palette='hls')
plt.show()


# In[509]:


## data exploration
# print(y.value_counts())
## salary data is skewed
y = salaryEngineering['Salary']
y.hist(bins=100)
plt.savefig("salary_bf")
plt.show()


# In[510]:


# Square root transformation & log transformation
y = np.log(y)
y = np.sqrt(y)
y.hist(bins=100)
plt.savefig("salary_af")
plt.show()


# In[498]:


salaryEngineering.groupby('Salary').mean()


# # Predicting with models

# In[545]:


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from math import sqrt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LassoCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import ElasticNet

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=1)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_test = pd.DataFrame(scaler.fit_transform(X_test), columns=X.columns)


# In[546]:


## KNN model
knn_model = KNeighborsRegressor(n_neighbors=3)
knn_model.fit(X_train, y_train)
train_preds = knn_model.predict(X_train)

error = []
for i in range(len(y_test)):
    if y_test.iloc[i] != train_preds[i]:
        error.append(np.absolute(y_test.iloc[i]-train_preds[i]))
plt.boxplot(error)
plt.savefig("knn_box")

print("MSE:", mean_squared_error(y_train, train_preds))
print("Score:", knn_model.score(X,y))


# In[554]:


## Linear Regression
reg = LinearRegression()
reg.fit(X_train, y_train)
print(pd.Series(reg.coef_, index = X.columns))
y_pred = reg.predict(X_test)

error = []
for i in range(len(y_test)):
    if y_test.iloc[i] != y_pred[i]:
        error.append(np.absolute(y_test.iloc[i]-y_pred[i]))
plt.boxplot(error)
plt.savefig("linear_box")

print("MSE:", mean_squared_error(y_test, y_pred))
print("Score:", reg.score(X_test, y_test))


# In[548]:


# Lasso with 5 fold cross-validation
LassoCV = LassoCV(cv=5, random_state=0, max_iter=10000)

# Fit model
LassoCV.fit(X_train, y_train)


# In[549]:


# Set best alpha
lasso_best = linear_model.Lasso(alpha=LassoCV.alpha_)
lasso_best.fit(X_train, y_train)

print(pd.Series(lasso_best.coef_, index = X.columns)) 


# In[550]:


lasso_pred = lasso_best.predict(X_test)
error = []
for i in range(len(y_test)):
    if y_test.iloc[i] != lasso_pred[i]:
        error.append(np.absolute(y_test.iloc[i]-lasso_pred[i]))
plt.boxplot(error)
plt.savefig("lasso_box")
print("MSE:", mean_squared_error(y_test, lasso_pred))
print("Score:", lasso_best.score(X,y))


# In[551]:


## Elastic net regression

model = ElasticNet(alpha=1.0, l1_ratio=0.5)
# define model evaluation method
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# force scores to be positive
scores = np.absolute(scores)
print('Mean MAE: %.6f (%.6f)' % (np.mean(scores), np.std(scores)))


# In[552]:


regr = ElasticNet(random_state=0)
regr.fit(X, y)
predX = regr.predict(X_test)
error = []
for i in range(len(y_test)):
    if y_test.iloc[i] != predX[i]:
        error.append(np.absolute(y_test.iloc[i]-predX[i]))
plt.boxplot(error)
plt.savefig("elastic_box")
print("MSE:", mean_squared_error(y_test, predX))


# In[553]:


## Ridge Regression
ridge = linear_model.Ridge(alpha = 0.01, normalize = True)
# Fit a ridge regression on the training data
ridge.fit(X_train, y_train)  
# Use this model to predict the test data
pred = ridge.predict(X_test)
# Print coefficients
print(pd.Series(ridge.coef_, index = X.columns)) 
# Calculate the test error
error = []
for i in range(len(y_test)):
    if y_test.iloc[i] != pred[i]:
        error.append(np.absolute(y_test.iloc[i]-pred[i]))
plt.boxplot(error)
plt.savefig("ridge_box")
print("MSE:", mean_squared_error(y_test, pred))  
print("Score:", ridge2.score(X,y))


# In[ ]:




