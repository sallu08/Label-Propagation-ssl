import tensorflow as tf
import numpy as np
from mnist_get_data import *
import keras
import scipy
from scipy.spatial.distance import pdist, squareform
from sklearn import preprocessing
from sklearn.utils import class_weight
from sklearn.utils.class_weight import compute_class_weight

x_train,y_train,x_test,y_test,label_index,x_train_labeled,y_train_labeled = get_data(no_of_samples=10) #10 samples for each class
input_shape = (28, 28, 1)
batch_size = 32
epochs = 50
num_classes = np.unique(y_train).size #10

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc')>0.99):
      print("\nReached 99% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (5,5), padding='same', activation='relu', input_shape=input_shape),
    tf.keras.layers.Conv2D(32, (5,5), padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(strides=(2,2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    
    tf.keras.layers.Dropout(0.5,name='ml'),   #store these values and performa Label Propagation
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy',
              metrics=['acc'],weighted_metrics=[])
history = model.fit(x_train_labeled, y_train_labeled,batch_size=batch_size,epochs=epochs,
                    validation_split=0.1,callbacks=[callbacks])
print('\n performance without Label Propagation')
test_loss, test_acc = model.evaluate(x_test, y_test)

#-----------------------------------------------------------------------------------------------------------------------------------
#Label Propagation Function


N=y_train.size #no of samples of train data
K=5 #no of k nearest neighbor
alpha=0.99
tolerance=1e-6

def label_propagation(feat):   
    norm = preprocessing.normalize(feat, norm='l2')
    norm=norm**3   #3=gamma
    distmat = squareform(pdist(norm,'euclidean'))
    neighbors = np.sort(np.argsort(distmat, axis=1)[:, 1:K+1])
    Af = np.zeros((N, N)) 
       
       
    for i, n in enumerate(neighbors):
            for _n in n:
                Af[i][_n]=distmat[i][_n]

    W = Af + Af.T
    W = W - scipy.sparse.diags(W.diagonal())
    S = W.sum(axis = 1)
    S[S==0] = 1
    D = np.array(1./ np.sqrt(S))
    D = scipy.sparse.diags(D.reshape(-1))
    Wn = D * W * D
    
    Z = np.zeros((N,num_classes))
    Ai = scipy.sparse.eye(Wn.shape[0]) - alpha * Wn
            
    for i in range(num_classes):
        cur_idx = label_index[np.where(y_train[label_index] ==i)]
        y = np.zeros((N,))
        y[cur_idx] = 1.0 / cur_idx.shape[0]
        f, _ = scipy.sparse.linalg.cg(Ai, y, tol=tolerance, maxiter=20) #changes made
        Z[:,i] = f
        
    Z[Z < 0] = 0 
    
    # Compute the weight for each instance based on the entropy (eq 11 from the paper)
    probs_l1 = preprocessing.normalize(Z, norm='l1')

    probs_l1[probs_l1 < tolerance] = tolerance
    entropy = scipy.stats.entropy(probs_l1.T)
    weights = 1 - entropy / np.log(num_classes)
    weights = weights / np.max(weights)
    
    p_labels = np.argmax(probs_l1,1)
    # Compute the accuracy of pseudolabels for statistical purposes
    correct_idx = (p_labels == y_train)
    acc = correct_idx.mean()
    print('\n label Propagation Accuracy,',acc)
    p_labels[label_index] = y_train[label_index]
    weights[label_index] = 1.0

    p_weights = weights
    
 
    class_weights = compute_class_weight(
                                        class_weight = "balanced",
                                        classes = np.unique(p_labels),
                                        y = p_labels                                                  
                                    )
    class_weight_dict= dict(zip(np.unique(p_labels), class_weights))
    p_labels = tf.one_hot(p_labels.astype(np.int32), depth=10)
            
    return p_labels,p_weights,class_weight_dict

#--------------------------------------------------------------------------------------------------------------------------------------
#Apply Label Propagation in Fit Function

layer_name1 = 'ml'
intermediate_layer_model1 = keras.Model(inputs=model.input,
                                            outputs=model.get_layer(layer_name1).output)

for i in range(25):
    feature1 = intermediate_layer_model1(x_train)
    p_label1,p_weight1,class_weights1=label_propagation(feature1)
    history0 = model.fit(x_train, p_label1, batch_size=batch_size,epochs=1,
                        validation_split=0.1,callbacks=[callbacks],sample_weight=np.array(p_weight1),
                        class_weight=class_weights1)
    loss=[]
    acc=[]
    val_loss=[]
    val_acc=[]

    loss.append(history0.history['loss'])
    acc.append(history0.history['acc'])
    val_loss.append(history0.history['val_loss'])
    val_acc.append(history0.history['val_acc'])
print('\n performance after Label Propagation')
test_loss1, test_acc1 = model.evaluate(x_test, y_test)
