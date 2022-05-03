import pandas as pd
import matplotlib as plt
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.datasets
from xgboost import XGBRegressor
import seaborn as sns
import pickle
from sklearn import metrics

house_price=sklearn.datasets.load_boston()
df=pd.DataFrame(house_price.data,columns=house_price.feature_names)
df['price']=house_price.target

x=df.drop(columns=['price'])
y=df['price']

x_train,x_test,y_train,y_test=train_test_split(x, y, test_size = 0.3, random_state = 2)
model=XGBRegressor()

model.fit(x_train,y_train)
pickle.dump(model,open('final_model.pkl','wb'))
load_model=pickle.load(open('final_model.pkl','rb'))