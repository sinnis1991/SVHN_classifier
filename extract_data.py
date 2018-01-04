import scipy.io
import os
import numpy as np



def extract(path):
    file_path = os.path.join(path,'train_32x32.mat')
    mat = scipy.io.loadmat(file_path)

    # print mat.keys()

    X = mat['X']
    y = mat['y']

    # print type(X)
    # print X.shape
    # print X.shape[3]

    # print X[:, :, :, 1000]
    X_ = np.adarray()
    

    for i in range(X.shape[3]):

      a = X_[i] & X[:,:,:,i]

      if not a.all() :
          print i
          print 'wrong!'
          break

    return X_





k = extract('./')
print k.shape