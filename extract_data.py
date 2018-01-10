import scipy.io
import os
import numpy as np



def extract(path,file):
    file_path = os.path.join(path,file)
    mat = scipy.io.loadmat(file_path)

    # print mat.keys()

    X = mat['X']
    y = mat['y']

    # print type(X)
    # print X.shape
    # print X.shape[3]

    # print X[:, :, :, 1000]
    X_ = np.empty((X.shape[3],X.shape[0],X.shape[1],X.shape[2]),dtype='uint8')

    for i in range(X.shape[3]):
        X_[i] = X[:,:,:,i]

    #check if the reshape is right
    for i in range(X.shape[3]):

      a = (X_[i] == X[:,:,:,i])

      if not a.all():
          print i
          print 'wrong!'
          break
    # print X[:,:,:,1]
    # print '---------'
    # print X_[1]
    # print X.dtype
    y_ = np.zeros((y.shape[0],10),dtype='uint8')
    # print y.shape
    # print y_.shape
    for i in range(y.shape[0]):
        index = y[i]
        if index == 10:
            index =0
        y_[i][index]=1

    # print y[:10]
    # print y_[:10]

    return X_,y_

def extract_batch(path,file,start,batch):

    file_path = os.path.join(path, file)
    mat = scipy.io.loadmat(file_path)

    X = mat['X']
    y = mat['y']

    X_ = np.empty((batch,X.shape[0],X.shape[1],X.shape[2]),dtype='uint8')

    for i in range(batch):
        X_[i] = X[:,:,:,start+i]

    y_ = np.zeros((batch, 10), dtype='uint8')
    for i in range(batch):
        index = y[start+i]
        if index == 10:
            index = 0
        y_[i][index] = 1

    return X_,y_

def extract_new(path,file):
    file_path = os.path.join(path, file)
    mat = scipy.io.loadmat(file_path)
    X = mat['X']
    y = mat['y']

    y_ = np.zeros((y.shape[0], 10), dtype='uint8')
    # print y.shape
    # print y_.shape
    for i in range(y.shape[0]):
        index = y[i]
        if index == 10:
            index = 0
        y_[i][index] = 1
    return X,y_

# m,n = extract_new('./','full_train_data.mat')
# print m.shape
# print n.shape