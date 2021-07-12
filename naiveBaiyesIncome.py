import csv 
import pandas as pd
import plotly.express as px 

df = pd.read_csv("./income.csv")

print(df.head())

from sklearn.model_selection import train_test_split
X = df[["age", "hours-per-week", "education-num", "capital-gain", "capital-loss"]]
y = df["income"]

x_train_1,x_test_1,y_train_1,y_test_1 = train_test_split(X,y,test_size = 0.25,random_state=42 )

#training the model with naive bayes 

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score 
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train_1= sc.fit_transform(x_train_1)
x_test_1 = sc.fit_transform(x_test_1)

model_1 = GaussianNB()
model_1.fit(x_train_1,y_train_1)

y_pred_1 = model_1.predict(x_test_1)

accuracy = accuracy_score(y_test_1,y_pred_1)

print(accuracy)

#We can see that we have an accuracy of 0.789 that is 78%

#---------------------------------------------------------
#Let's see if using logistic regression what would be the accuracy 
from sklearn.model_selection import train_test_split
X = df[["age", "hours-per-week", "education-num", "capital-gain", "capital-loss"]]
y = df["income"]

x_train_2,x_test_2,y_train_2,y_test_2 = train_test_split(X,y,test_size = 0.25,random_state=42 )
#training the model with logistic regression 

from sklearn.metrics import accuracy_score 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression 

sc = StandardScaler()
x_train_2= sc.fit_transform(x_train_2)
x_test_2 = sc.fit_transform(x_test_2)

model_2 = LogisticRegression(random_state = 0)
model_2.fit(x_train_2,y_train_2)

y_pred_2 = model_2.predict(x_test_2)

accuracy = accuracy_score(y_test_2,y_pred_2)

print(accuracy)

#By using Logistic Regression we get the accuracy of 0.811 that is around 81%
#for this dataset logistic regression worked better than that of naive bayes

#Conclusion: Logistic Regression outperforms naive bayes in this dataset as : 
# not all features contribute individualy to outcome 












