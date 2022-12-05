import tensorflow as tf
import numpy as np
from random import *
x = randint(1, 6)

def get_data(no_of_samples): #where n is number of samples you want for each digit

    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data() 
    x_train=x_train[(x-1)*10000:x*10000] #randomly select section of 10000 images 
    y_train=y_train[(x-1)*10000:x*10000]

    
    ind_arr=[]
    
    for i in range(10):  #value is 10 becouse we have label 0-9
        count=0
        for ind,val in enumerate(y_train):    
            if(val==i):
                if(count<no_of_samples):
                    ind_arr=np.append(ind_arr,ind)
                    count=count+1
         

    label_ind = ind_arr.astype(int) 
    x_train_labeled=x_train[label_ind]
    y_train_labeled=y_train[label_ind]

    x_train_labeled=x_train_labeled.reshape(x_train_labeled.shape[0], x_train_labeled.shape[1], x_train_labeled.shape[2], 1)
    x_train_labeled=x_train_labeled / 255.0

    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
    x_test=x_test/255.0

    x_train=x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    x_train=x_train / 255.0

    #one-hot encoding
    y_train_labeled = tf.one_hot(y_train_labeled.astype(np.int32), depth=10)
    y_test = tf.one_hot(y_test.astype(np.int32), depth=10)

    
    return x_train,y_train,x_test,y_test,label_ind,x_train_labeled,y_train_labeled
