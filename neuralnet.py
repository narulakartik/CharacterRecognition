# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 15:30:30 2020

@author: narul
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 14:03:50 2020

@author: narul
"""

import numpy as np
import sys
import time as time
starttime=time.time()

def forward(x,y,alpha, beta, classes):
    
    xt=np.transpose(x)
    a=alpha@xt
    z=sigma(a)
    zt=np.transpose(z)

    zt=np.insert(zt,0,1,1)

    zt1=np.transpose(zt)
    b=beta@zt1
    b=np.transpose(b)
    y_predicted=np.zeros((len(x), classes))

    for i in range(len(x)):
       y_predicted[i]=softmax(b[i])
    
    j=crossentropy(y,y_predicted)
    return a,zt,b,y_predicted,j

def softmax(b):
 
 
 return np.exp(b) / np.sum(np.exp(b))   



def sigma(a):
  s=1/(1+np.exp(-a))
  return s


def dzda(z):
    dz_da=np.diag(z[0]*(1-z[0]))
    #dz_da=np.zeros((len(z[0]),len(z[0])))
    #for i in range(len(z[0])):
        
     #           dz_da[i][i]=z[0][i]*(1-z[0][i])
   
         
    return dz_da    



def crossentropy(y,y_predicted):
    
    
    cross=np.log(y_predicted[0][int(y)])    
   
    return -cross


def hotvector(y, classes):
    h=np.zeros((len(y), classes ))
    for i in range(len(y)):
        
            
        h[i][int(y[i])]=1
        
    return h        

#def y_predicted_backward(y, y_predicted):
 #   gy_predicted=-y/y_predicted
  #  return gy_predicted

#def b_backward(b,y_predicted, classes):   
 #   dbdy_predicted=np.empty((classes, classes))
  #  for i in range(classes):
   #     for j in range(classes):
    #        if i==j:
     #           dbdy_predicted[i][j]=y_predicted[0][i]*(1-y_predicted[0][i])
      #      else:
       #         dbdy_predicted[i][j]=y_predicted[0][i]*(-y_predicted[0][j])
                
    #return dbdy_predicted 


def backward(x,a,z,b,y_predicted,j ,classes,h):
    #djdyp=y_predicted_backward(h,y_predicted)
    #dypdb=b_backward(b,y_predicted,classes)
    
    #dypdb_t=np.transpose(dypdb)
    #djdb=djdyp@dypdb_t
    
     
    djfdb=djdb(h,y_predicted)
    djdb_t=np.transpose(djfdb)
    
    
    
    djdb_t=np.transpose(djfdb)
    
    djdbeta=djdb_t@z
    beta_dash=np.delete(beta,0,1)
    z_dash=np.delete(z,0,1)
    djdz=djfdb@beta_dash
    #djdz_t=np.transpose(djdz)
    dz_da=dzda(z_dash)

    djda=djdz@dz_da
    djda_t=np.transpose(djda)
    djdalpha=djda_t*x
    return djdalpha, djdbeta


    

def sgd(x,y,alpha,beta,classes,gamma,h):
    a,z,b,y_predicted,j=forward(x,y,alpha,beta,classes)
    djdalpha, djdbeta=backward(x,a,z,b,y_predicted,j,classes,h)
    alpha=alpha-djdalpha*gamma
    beta=beta-djdbeta*gamma
    
    return alpha, beta



def predict(x, y, alpha, beta, classes):
   a,zt,b,y_predicted,j=forward(x,y,alpha, beta, classes)    
   m=max(y_predicted[0])
   return np.where(y_predicted[0]==m)[0][0]

def djdb(h,y_predicted):
    dj_db=y_predicted-h
    return dj_db





train_file=sys.argv[1]
test_file=sys.argv[2]
train_label=sys.argv[3]
test_label=sys.argv[4]
metrics_train=sys.argv[5]
metrics_test=sys.argv[6]
epoch=int(sys.argv[7])
hidden_units=int(sys.argv[8])
init_flag=int(sys.argv[9])
gamma=float(sys.argv[10])

metrics_train=open(metrics_train, 'w')
metrics_test=open(metrics_test, 'w')
train_label=open(train_label, 'w')
test_label=open(test_label, 'w')
classes=10

train=np.genfromtxt(train_file, delimiter=',')
test=np.genfromtxt(test_file, delimiter=',')

y=np.array(train[:,0])
x=np.delete(train,0,1)
x=np.insert(x,0,1,1)
xs=np.split(x,len(x),axis=0)


y_test=np.array(test[:,0])
x_test=np.delete(test,0,1)
x_test=np.insert(x_test,0,1,1)
xt=np.split(x_test,len(x_test),axis=0)


h=hotvector(y,classes)

#initializing weights

if init_flag==1:
    alpha=np.random.random_sample((hidden_units,x.shape[1]))*0.1+((np.random.random_sample((hidden_units,x.shape[1])))*0.1)-0.1
    beta=np.random.random_sample((classes,hidden_units+1))*0.1+((np.random.random_sample((classes,hidden_units+1)))*0.1)-0.1
elif init_flag==2:
    alpha=np.zeros((hidden_units, x.shape[1]))
    beta=np.zeros((classes, hidden_units+1))


#training
for i in range(epoch):
    cross_entropy_train=0
    train_error=0
    for j in range(len(xs)):
        
        alpha, beta=sgd(xs[j],y[j],alpha, beta, classes, 0.1,h[j])
         
    
    for j in range(len(xs)):
         a,z,b,y_predicted,e=forward(xs[j],y[j],alpha,beta,classes)    
         cross_entropy_train+=e 
        
    
    cross_entropy_test=0
    for j in range(len(xt)):
        a,z,b,y_predicted,e=forward(xt[j],y_test[j],alpha,beta,classes)    
        cross_entropy_test+=e
    cross_entropy_test=cross_entropy_test/len(xt)   
    cross_entropy_train=cross_entropy_train/len(xs)    
    #metrics.write("epoch=")
    #metrics.write(str(i+1))
    #metrics.write(" ")
    #metrics.write("crossentropy(train): ")
    metrics_train.write(str(cross_entropy_train))
   
    metrics_train.write("\n")
    
    #metrics.write("epoch=")
    #metrics.write(str(i+1))
    #metrics.write(" ")
    #metrics.write("crossentropy(test): ")
    metrics_test.write(str(cross_entropy_test))
    
    metrics_test.write("\n")
    
    

#train_prediction and error

train_error=0
#labels_train
for j in range(len(xs)):
    a=predict(xs[j], y[j], alpha, beta, classes)
    train_label.write(str(a))
    train_label.write("\n")
    if a!=int(y[j]):
        train_error+=1
train_error/=len(xs)
    
#metrics.write("error(train): ")
#metrics.write(str(train_error))
#metrics.write("\n")

#test prediction and error
test_error=0
for j in range(len(xt)):
    a=predict(xt[j], y_test[j], alpha, beta, classes)
    test_label.write(str(a))
    test_label.write("\n")
    
    if a!=y_test[j]:
        test_error+=1
test_error/=len(xt)


#metrics.write("error(test): ")
#metrics.write(str(test_error))


endtime=time.time()
print(endtime-starttime)
#print(beta)
#print(alpha)
