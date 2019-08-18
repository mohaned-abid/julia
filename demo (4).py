import numpy as np 
import pandas as pd

import pandas as pd 
from skimage.io import imread
from skimage.transform import resize
import numpy as np
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense,InputLayer,MaxPooling2D,Dropout,Conv2D,Flatten
from keras.optimizers import SGD
#importing and preprocessing trainning data
df=pd.read_csv('D:\Mohanned 10\Downloads\julia\kk.csv')
map1={'g': 0, 'Q': 1, 'k': 2, 'N': 3, 'p': 4, 'L': 5, 'U': 6, 'X': 7, 'J': 8, 'i': 9, 'H': 10, 'r': 11, 'q': 12, 'v': 13, '6': 14, 'M': 15, 'x': 16, 'Y': 17, 'o': 18, 'P': 19, 'a': 20, '9': 21, 'O': 22, 'K': 23, 'h': 24, 'n': 25, 't': 26, 'E': 27, 'G': 28, 'C': 29, 'Z': 30, 'm': 31, 'z': 32, 's': 33, 'f': 34, 'l': 35, 'S': 36, 'd': 37, '8': 38, 'y': 39, 'u': 40, 'A': 41, 'b': 42, '5': 43, 'c': 44, 'I': 45, '3': 46, 'w': 47, 'T': 48, 'e': 49, '1': 50, '0': 51, '2': 52, '7': 53, 'j': 54, '4': 55, 'F': 56, 'B': 57, 'W': 58, 'V': 59, 'R': 60, 'D': 61}
df['Class']=df['Class'].map(map1)
trainy=df[['Class']]
tt=np.zeros(6283)
for (ind,i) in enumerate(trainy.values):
    tt[ind]=i
trainx=np.zeros((df.shape[0],1024))
for (ind,idd) in enumerate(df['ID']):
                              filename='D:/Mohanned 10/Downloads/julia/train/{0}.Bmp'.format(idd)
                              image=imread(filename,as_gray=True)
                              image=resize( image, (32,32) )
                              trainx[ind,:]=np.reshape(image,(1,1024))
trainx=trainx.reshape((trainx.shape[0], 32, 32, 1))
trainy = to_categorical(trainy,62)
#importing and trainning the model
model = Sequential()
model.add(Conv2D(32, (3, 3), padding="same",input_shape=(32,32,1)))
model.add(Activation("relu"))
model.add(Conv2D(32, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
 
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation("relu"))
model.add(Dropout(0.5))
 
model.add(Dense(62))
model.add(Activation("softmax"))
opt=SGD(lr=0.01,momentum=0.9)
model.compile(loss='categorical_crossentropy', optimizer=opt)
model.fit(trainx, trainy, epochs=100,verbose = 1)
#importing and predicting test data
df1=pd.read_csv('D:\Mohanned 10\Downloads\julia\sampleSubmission.csv')
testx=np.zeros((df1.shape[0],1024))
for (ind,idd) in enumerate(df1['ID']):
                              filename='D:/Mohanned 10/Downloads/julia/test/{0}.Bmp'.format(idd)
                              image=imread(filename,as_gray=True)
                              image=resize( image, (32,32) )
                              testx[ind,:]=np.reshape(image,(1,1024))
testx=testx.reshape((testx.shape[0], 32, 32, 1))
l=model.predict_classes(testx,verbose=1)
df1['Class'] = l
map2={0: 'g', 1: 'Q', 2: 'k', 3: 'N', 4: 'p', 5: 'L', 6: 'U', 7: 'X', 8: 'J', 9: 'i', 10: 'H', 11: 'r', 12: 'q', 13: 'v', 14: '6', 15: 'M', 16: 'x', 17: 'Y', 18: 'o', 19: 'P', 20: 'a', 21: '9', 22: 'O', 23: 'K', 24: 'h', 25: 'n', 26: 't', 27: 'E', 28: 'G', 29: 'C', 30: 'Z', 31: 'm', 32: 'z', 33: 's', 34: 'f', 35: 'l', 36: 'S', 37: 'd', 38: '8', 39: 'y', 40: 'u', 41: 'A', 42: 'b', 43: '5', 44: 'c', 45: 'I', 46: '3', 47: 'w', 48: 'T', 49: 'e', 50: '1', 51: '0', 52: '2', 53: '7', 54: 'j', 55: '4', 56: 'F', 57: 'B', 58: 'W', 59: 'V', 60: 'R', 61: 'D'}
df1['Class']=l
df1['Class']=df1['Class'].map(map2)
df1.to_csv('D:/Mohanned 10/Downloads/julia/sub/out.csv')
