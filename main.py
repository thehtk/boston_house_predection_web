import pandas as pd
import numpy as np
import warnings
import pickle
warnings.filterwarnings("ignore")
df =pd.read_csv("C:\\Users\\Hp\\OneDrive\\Projects\\boston\\boston.csv")
X=df[['nox','rm','dis','ptratio','lstat']]
y=df['medv']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

pickle.dump(regressor,open('model.pkl','wb'))


model = pickle.load(open('model.pkl','rb'))
