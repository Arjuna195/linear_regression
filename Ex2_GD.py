'''THIS IS A DIRECT GD DONE BASED ON THE LOOPING Y=MX+C AND GD FORMULA SAME LIKE EXCEL EXCEPT NOT CALCULATING COST FUNCTION DIRECTLY BUT CALCULATED LOSS '''
import pandas as pd
import numpy as np
import seaborn as sb
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

df = pd.read_csv('C:/Users/Nagarjuna.Reddy/Downloads/Advertising.csv')
#print(df.head())

#print(df.describe())
x = df['TV']
y = df['Sales']


prediction=[]
x_train,x_test = x[:160],x[160:]
y_train,y_test = y[:160],y[160:]

'''for val in x_test:
    y_pred = m*val+c
    prediction.append(y_pred)'''

#print(len(prediction))
print(x_train.shape)
print(y_train.shape)
print(y_test.shape)

#GD
epoch = 1000
lr = 0.00001
n= len(x_train)
m=0
c=0
def gd(lr,epoch,m,c,x,y):
    for i in range(epoch):
        n = len(x)
        y_pred = m*x+c
        loss_m =-2/n*np.sum((y-y_pred)*x)
        loss_c =-2/n*np.sum(y-y_pred)

        m = m-lr*loss_m
        c = c-lr*loss_c
    return m,c

new_m,new_c=gd(lr,epoch,m,c,x_train,y_train)
print(new_m,new_c)
# pred = new_m*x_test+new_c  # DIRECT CODE ALSO WORKS
# or

pred = []
for val in x_test:
    y_pred = new_m*val+new_c
    pred.append(y_pred)

#print(pred)
r2 = r2_score(y_test,pred)
mse = mean_squared_error(y_test,pred)
print(f'R2 score of the model is {r2},Mean Square Error of the model is {mse}')

plt.scatter(x_test,y_test)
plt.plot(x_test,pred)
plt.show()
