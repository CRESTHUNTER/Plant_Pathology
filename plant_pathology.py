import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, InputLayer, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np
from scipy.ndimage.interpolation import zoom
import matplotlib.pyplot as plt
import cv2
num=list(range(0,1821))
names=[]
names_test=[]

from keras import regularizers
from keras.applications import resnet50
from keras.applications import vgg16
from keras import optimizers
import tensorflow as tf

def get_labels(path):
    df=pd.read_csv(path)
    labels=[]
    for i in range(0,len(df)):
        temp=[]
        temp.append(df['healthy'][i])
        temp.append(df['multiple_diseases'][i])
        temp.append(df['rust'][i])
        temp.append(df['scab'][i])
        temp=np.array(temp)
        labels.append(temp)
    return df, labels

num_list=list(range(0,1821))

names_comp=[]
for n,i in enumerate(num_list):
    names_comp.append(Image.open('Test_'+str(i)+'.jpg'))
names_comp_array=[]

for n,i in enumerate(names_comp):
    names_comp_array.append(np.array(i))
    print(n,end='\r')
print('Train images converting  Done')

names_comp_test=[]
for n,i in enumerate(num_list):
    names_comp_test.append(Image.open('Test_'+str(i)+'.jpg'))
names_comp_array_test=[] 
for n,i in enumerate(names_comp_test):
    names_comp_array_test.append(np.array(i))
    print(n,end='\r')
print('Test images converting Done')


labels_df, labels=get_labels(path='train.csv')

#Data Augmentation


#ROTATION
def rotate_images(X_imgs):
    X_rotate=[]
    for n,i in enumerate(X_imgs):
        X_rotate.append(np.rot90(i, 1))
    return X_rotate
	
rotated_imgs_90  = rotate_images(names_comp_array)
rotated_imgs_180 = rotate_images(rotated_imgs_90)
rotated_imgs_270 = rotate_images(rotated_imgs_180)

import itertools
names_comp_array_final=[]
names_comp_array_final = itertools.chain(names_comp_array, rotated_imgs_90, rotated_imgs_180, rotated_imgs_270)
names_comp_array_final=list(names_comp_array_final)
labels_final=[]
labels_final = itertools.chain(labels,labels,labels,labels)
labels_final=list(labels_final)

#ADDING NOISE

def add_salt_pepper_noise(X_imgs):
    # Need to produce a copy as to not modify the original image
    X_imgs_copy = X_imgs.copy()
    row, col, _ = X_imgs_copy[0].shape
    salt_vs_pepper = 0.2
    amount = 0.004
    num_salt = np.ceil(amount * X_imgs_copy[0].size * salt_vs_pepper)
    num_pepper = np.ceil(amount * X_imgs_copy[0].size * (1.0 - salt_vs_pepper))
    
    for X_img in X_imgs_copy:
        # Add Salt noise
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in X_img.shape]
        X_img[coords[0], coords[1], :] = 1

        # Add Pepper noise
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in X_img.shape]
        X_img[coords[0], coords[1], :] = 0
    return X_imgs_copy
  
salt_pepper_noise_imgs = add_salt_pepper_noise(names_comp_array_final)
names_comp_array_ultimate=[]
names_comp_array_ultimate =itertools.chain(names_comp_array_final, salt_pepper_noise_imgs)
names_comp_array_ultimate=list(names_comp_array_ultimate)
labels_ultimate=[]
labels_ultimate = itertools.chain(labels_final,labels_final)
labels_ultimate = list(labels_ultimate)



model = Sequential()
# model.add(InputLayer(input_shape=(input_shape,)))
# model.add(vgg16.VGG16( , weights='imagenet',include_top=False)),input_shape=(224,224,3)
model.add(Conv2D(32, kernel_size = (3, 3),input_shape=(224,224,3), activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3,3), activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3,3), activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(96, kernel_size=(3,3), activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3,3), activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=(3,3), activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(4, activation = 'softmax'))

# model.compile(loss='binary_crossentropy',
#               optimizer=optimizers.RMSprop(lr=1e-4),
#               metrics=['accuracy'])

# model.summary()



model.compile(optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])

history=model.fit(np.stack(names_comp_array_ultimate, axis=0), np.stack(labels_ultimate,axis=0), batch_size=50,callbacks=[lrs], epochs=10,validation_split=0.15,)

test_df=pd.read_csv('test.csv')
result=model.predict(np.stack(names_comp_array_test,axis=0))

for n in range(0,len(test_df)):
    test_df['healthy'][n] = result[n][0]
    test_df['multiple_diseases'][n]=result[n][1]
    test_df['rust'][n]=result[n][2]
    test_df['scab'][n]=result[n][3]
    print(n, end='\r')

test_df.to_excel('test.csv')