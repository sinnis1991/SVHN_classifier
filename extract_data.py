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





m,n = extract('./','train_32x32.mat')
print m.shape
print n.shape