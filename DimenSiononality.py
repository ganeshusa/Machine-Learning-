import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

data = pd.read_csv('train.csv')
print(data.shape)
print(data.head())

X = data.iloc[:, :-1]
y = data.iloc[:, -1]
print(X.shape)
print(y.shape)

mu=X.mean(axis=0)
sig=X.std(axis=0)
X=(X-mu)/sig

best_features= SelectKBest()
fit= best_features.fit(X,y)
print(fit.scores_)

dfScores=pd.DataFrame(best_features.scores_)
dfColumns=pd.DataFrame(X.columns)
featureScores=pd.concat([dfColumns,dfScores],axis=1)
featureScores.columns=['Features','Scores']
featureScores.sort_values(by='Scores',ascending=False,inplace=True)

top_10_features=list(featureScores[:10]['Features'].values)
print(top_10_features)

model= LogisticRegression()
model.fit(X,y)
scores = cross_val_score(model,X,y,cv=10,scoring='accuracy')
print(scores.mean())
scores = cross_val_score(model,X[top_10_features],y,cv=10,scoring='accuracy')
print(scores.mean())