import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import pickle

#read data
dataset=pd.read_csv('Cellphone.csv')
print(dataset.head())

# x1=np.array(dataset['weight'])
# x2=np.array(dataset['resoloution'])
# x3=np.array(dataset['cpu core'])
# x4=np.array(dataset['cpu freq'])
# x5=np.array(dataset['internal mem'])
# x6=np.array(dataset['ram'])
# x7=np.array(dataset['RearCam'])
# x8=np.array(dataset['Front_Cam'])
# x9=np.array(dataset['battery'])
# x10=np.array(dataset['ppi'])

dataset2=dataset[['resoloution','ppi','cpu core','cpu freq','internal mem','ram','RearCam','Front_Cam','battery']]
print(dataset2)
x = np.array(dataset.drop(['Product_id','Price','Sale','thickness','weight'],1), dtype=np.int64)
y=np.array(dataset['Price'],dtype=np.int64)
x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(x,y,test_size=0.1)


best=0
for _ in range(1000):
    x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(x,y,test_size=0.1)

    #use the linearRegression module
    linear=linear_model.LinearRegression()

    #minimizing error(gradient decent)
    linear.fit(x_train,y_train)

    #test the accuracy of the error
    acc=linear.score(x_test,y_test)

    #printing out some values for verification
    #test model accuracy level


    if acc>best:
        best=acc
        with open('studentmodel.pickle','wb') as f:
           pickle.dump(linear,f)

savedmodel=open('studentmodel.pickle', 'rb')
newlinear=pickle.load(savedmodel)
mine=[[5,401,4,2,16,2,16,8,2500]]
mine=np.array(mine)
#predict the outcome of your value(s)
predictions=newlinear.predict(x_test)

#loop through prediction to see if your data is corresponding well
for x in range(len(predictions)):
    print(predictions[x],x_test[x],y_test[x])

