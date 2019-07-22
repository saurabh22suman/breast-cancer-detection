# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 11:59:11 2019

@author: Saurabh
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()


df_cancer = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns = np.append(cancer['feature_names'], ['target']))


# Visualizing the data 

sns.pairplot(df_cancer, hue='target', vars = ['mean radius', 'mean texture', 'mean area', 'mean perimeter', 'mean smoothness'])


plt.figure(figsize=(20,10))
sns.heatmap(df_cancer.corr(),annot=True)

X = df_cancer.drop(['target'], axis=1)
y= df_cancer['target']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix

svc_model= SVC()
svc_model.fit(X_train,y_train)

y_predict = svc_model.predict(X_test)

cm = confusion_matrix(y_test,y_predict)

sns.heatmap(cm,annot=True)

# Improving the model

X_train_scaled = (X_train - X_train.min())/(X_train - X_train.min()).max()

X_test_scaled = (X_test - X_test.min())/(X_test - X_test.min()).max()

svc_model.fit(X_train_scaled,y_train)


y_predict_scaled = svc_model.predict(X_test_scaled)

cm_scaled = confusion_matrix(y_test,y_predict_scaled)

sns.heatmap(cm_scaled, annot = True)


print(classification_report(y_test,y_predict_scaled))



param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']} 
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=4)
grid.fit(X_train_scaled,y_train)

# grid.best_params_  (for checking the best C value and gamma value)
grid_predictions = grid.predict(X_test_scaled)

cm_gamma = confusion_matrix(y_test,grid_predictions)

sns.heatmap(cm_gamma , annot=True)
