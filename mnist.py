# -*- coding: utf-8 -*-
"""
@author: chandu
"""
import os
import struct
import numpy as np
from neuralnet import NeuralNetMLP

def load_mnist(path, kind='train'):
    # load mnist data from path
    labels_path = os.path.join(path,'%s-labels-idx1-ubyte'%kind)
    images_path = os.path.join(path,'%s-images-idx3-ubyte'%kind)
    with open(labels_path,'rb') as lbpath:
        magic,n = struct.unpack('>II',  lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
        
    with open(images_path,'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
        
    return images, labels

X_train, y_train = load_mnist('/Users/chandu/Documents/UTD S2/ML Projects/', kind = 'train')
X_test, y_test = load_mnist('/Users/chandu/Documents/UTD S2/ML Projects/', kind = 't10k')

nn = NeuralNetMLP(n_output=10, n_features=X_train.shape[1], n_hidden=50, 
                  l2=0.1, l1=0.0, epochs=1000, eta=0.001, alpha=0.001, 
                  decrease_const=0.00001, shuffle=True, minibatches=50, random_state=1)

nn.fit(X_train, y_train, print_progress=True)

y_train_pred = nn.predict(X_train)
acc = np.sum(y_train == y_train_pred, axis=0)/X_train.shape[0]
print('Training accuracy: %.2f%%' %(acc*100))

y_test_pred = nn.predict(X_test)
acc = np.sum(y_test == y_test_pred, axis=0)/X_test.shape[0]
print('Test accuracy: %.2f%%' %(acc*100))