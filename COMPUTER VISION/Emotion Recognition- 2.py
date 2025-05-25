#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:


import cv2
import matplotlib.pyplot as plt
import pandas as pd
import mediapipe as mp 
import numpy as np
import itertools
import tensorflow as tf
from tensorflow.keras import activations
from tensorflow.keras import losses

from tensorflow.keras.regularizers import l2, l1
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import RMSprop, SGD, Adagrad, Adam
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization ,Activation 
import keras 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
from tensorflow.keras.layers import Input, MaxPooling2D, Flatten, Dense,InputLayer ,Conv2D ,MaxPool2D
from tensorflow.keras import regularizers


# In[ ]:


mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
LEFT_EYE = list(set(itertools.chain(*mp_face_mesh.FACEMESH_LEFT_EYE)))
RIGHT_EYE = list(set(itertools.chain(*mp_face_mesh.FACEMESH_RIGHT_EYE)))
LEFT_EYEBROW = list(set(itertools.chain(*mp_face_mesh.FACEMESH_LEFT_EYEBROW)))
RIGHT_EYEBROW = list(set(itertools.chain(*mp_face_mesh.FACEMESH_RIGHT_EYEBROW)))
LIPS = list(set(itertools.chain(*mp_face_mesh.FACEMESH_LIPS)))
CONTOURS = list(set(itertools.chain(*mp_face_mesh.FACEMESH_CONTOURS)))
OTHER = [1]
face_mesh = mp_face_mesh.FaceMesh(  
                                    static_image_mode=True,
                                    max_num_faces=1,
                                    refine_landmarks=True,
                                    min_detection_confidence=0.5)
img=cv2.imread('/kaggle/input/emotion-detection-fer/test/happy/im1023.png')
#img=cv2.imread(df['image_path'][1])
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (500, 500)) # Any size, just for visualization
img = cv2.GaussianBlur(img, (3, 3), cv2.BORDER_DEFAULT)
img_shape = img.shape[0]
print("sahpe of the image it should be 500",img_shape)
results = face_mesh.process(img)
print("the results",results)
annotated_image = img.copy()
shape = [(lmk.x, lmk.y, lmk.z) for i, lmk in enumerate(results.multi_face_landmarks[0].landmark)]

shape = np.array(shape)
print(shape[1])
print("======")
print(len(shape))  #x,y,z #this after normalization
shape = shape[LEFT_EYE + RIGHT_EYE + LEFT_EYEBROW + RIGHT_EYEBROW + LIPS + OTHER]
print("the shape of the it ",shape.shape)
for lmk in shape:
    cv2.circle(annotated_image, (int(lmk[0] * img_shape), int(lmk[1] * img_shape)), 2, (0, 0, 255))
plt.imshow(annotated_image, interpolation='nearest')


# In[ ]:


import os

PTH = '/kaggle/input/emotion-detection-fer'
train_path = f'{PTH}/train'  
test_path = f'{PTH}/test'

def euc2d(a, b): # in 2d dimension
    return np.sqrt( (a[0]-b[0])*(a[0]-b[0]) + (a[1]-b[1])*(a[1]-b[1]) )

def euc3d(a, b):#in third dimension
    return np.sqrt( (a[0]-b[0])*(a[0]-b[0]) + (a[1]-b[1])*(a[1]-b[1]) + (a[2]-b[2])*(a[2]-b[2]) )

def prepare_csv(path, mode, face_mesh):
    
    emotions = os.listdir(path) #all directores of the train or the test the emotion exactly
    
    df = pd.DataFrame({
    #he choose the 92 becouse the range(5) => 0-4 so it will be acually a 5 number 
    # *2  is to be x and y 
    }, columns = [f"{i}" for i in range(92 * 2)] + ["y"])
    
    
    for i, emotion in enumerate(emotions): #also i we make it as labels
        images = os.listdir(f'{path}/{emotion}')
        for image in images:
            #pre process of the image
            img = cv2.imread(f"{path}/{emotion}/{image}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.GaussianBlur(img, (3, 3), cv2.BORDER_DEFAULT)
            
            results = face_mesh.process(img)
        
            if results.multi_face_landmarks:
        
                shape = [(lmk.x, lmk.y, lmk.z) for lmk in results.multi_face_landmarks[0].landmark]
                shape = np.array(shape) #all thing
                nose = shape[1] #the nose of the shape
                shape = shape[LEFT_EYE + RIGHT_EYE + LEFT_EYEBROW + RIGHT_EYEBROW + LIPS] 
                #the interested indexes from the landmark 

                distances2d = [round(euc2d(nose, x), 6) for x in shape]
                distances3d = [round(euc3d(nose, x), 6) for x in shape]

                df.loc[len(df)] = distances2d + distances3d + [i] #need more inveseitigation
            
    df.to_csv(f'{mode}.csv', index=False)


prepare_csv(train_path, 'train', face_mesh)
prepare_csv(test_path, 'test', face_mesh)


# In[ ]:


df_train=pd.read_csv('/kaggle/working/train.csv')
df_test=pd.read_csv('/kaggle/working/test.csv')


# In[ ]:


df_train.head()


# In[ ]:


df_test.head()


# In[ ]:


df_train.shape


# In[ ]:


df_test.shape


# In[ ]:


datagen = ImageDataGenerator(
                        brightness_range=[0.1, 1.5],
                        rotation_range=5, # rotate the image 30 degrees
                        width_shift_range=0.1, # Shift the pic width by a
                        height_shift_range=0.1, # Shift the pic height by
                        rescale=1./255, # Rescale the image by normalzing
                        shear_range=0.2, # Shear means cutting away part o
                        zoom_range=0.3, # Zoom in by 20% max
                        horizontal_flip=True, # Allow horizontal flipping
                        fill_mode='nearest', # Fill in missing pixels with
                        validation_split=0.3

)


# In[ ]:


batch_size = 32
image_size = (48, 48)  # Target image size

train_generator = datagen.flow_from_directory(
    '/kaggle/input/emotion-detection-fer/train/',
    target_size=image_size,
    batch_size=batch_size,
    color_mode = "rgb",
    class_mode='categorical',
    subset='training'       # 'training' or 'validation'
    
)
validation_generator = datagen.flow_from_directory(
    '/kaggle/input/emotion-detection-fer/train',
    target_size=image_size,
    batch_size=batch_size,
    color_mode = "rgb",
    class_mode='categorical',
    subset='validation'  
)


# In[ ]:


train_n = train_generator.n
val_n = validation_generator.n
image_shape=(48,48,3)
model = Sequential()
model.add(InputLayer(input_shape=image_shape))

# 1 CONVOLUTIONAL LAYER
model.add(Conv2D(filters=16, kernel_size=(3,3), padding="same"))
model.add(BatchNormalization())
model.add(Activation(activation=activations.swish))

model.add(Conv2D(filters=32, kernel_size=(3,3), padding="same"))
model.add(BatchNormalization())
model.add(Activation(activation=activations.swish))

model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same"))
model.add(BatchNormalization())
model.add(Activation(activation=activations.swish))

model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same"))
model.add(BatchNormalization())
model.add(Activation(activation=activations.swish))

# POOLING LAYER
model.add(MaxPool2D(pool_size=(3, 3)))

# 2 CONVOLUTIONAL LAYER
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same"))
model.add(BatchNormalization())
model.add(Activation(activation=activations.swish))
# POOLING LAYER
model.add(MaxPool2D(pool_size=(3, 3)))
model.add(Flatten())

model.add(Dense(512, activation='swish'))
model.add(Dense(256, activation='swish'))
model.add(Dense(7, activation='softmax'))
model.compile(loss = losses.CategoricalCrossentropy(),
                        optimizer = Adam(learning_rate=0.0003),
                        metrics = ['accuracy'])
results = model.fit(train_generator, epochs=100,
                    steps_per_epoch=train_n//batch_size,
                    validation_data=validation_generator,
                    validation_steps=val_n//batch_size)
model.save('best_fer_model.h5')


# In[ ]:


train_score = model.evaluate(train_generator, verbose= 1)
valid_score = model.evaluate(validation_generator,  verbose= 1)
# test_score = model.evaluate(test_gen, steps= test_steps, verbose= 1)
print("Train Loss: ", train_score[0])
print("Train Accuracy: ", train_score[1])
print('-' * 40)
print("Validation Loss: ", valid_score[0])
print("Validation Accuracy: ", valid_score[1])
print('-' * 40)


# In[ ]:


def plot_accuracy_validation(history):
    fig , ax = plt.subplots(1,2)
    train_acc = results.history['accuracy']
    train_loss = results.history['loss']
    fig.set_size_inches(12,4)

    ax[0].plot(results.history['accuracy'])
    ax[0].plot(results.history['val_accuracy'])
    ax[0].set_title('Training Accuracy vs Validation Accuracy')
    ax[0].set_ylabel('Accuracy')
    ax[0].set_xlabel('Epoch')
    ax[0].legend(['Train', 'Validation'], loc='upper left')

    ax[1].plot(results.history['loss'])
    ax[1].plot(results.history['val_loss'])
    ax[1].set_title('Training Loss vs Validation Loss')
    ax[1].set_ylabel('Loss')
    ax[1].set_xlabel('Epoch')
    ax[1].legend(['Train', 'Validation'], loc='upper left')

    plt.show()
plot_accuracy_validation(results)


# In[ ]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
Y_pred = model.predict_generator(validation_generator, val_n // batch_size+1)
y_pred = np.argmax(Y_pred, axis=1)
CLASS_LABELS  = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sadness', "Surprise"]
cm_data = confusion_matrix(validation_generator.classes, y_pred)
cm = pd.DataFrame(cm_data, columns=CLASS_LABELS, index = CLASS_LABELS)
cm.index.name = 'Actual'
cm.columns.name = 'Predicted'
plt.figure(figsize = (15,10))
plt.title('Confusion Matrix', fontsize = 20)
sns.set(font_scale=1.2)
ax = sns.heatmap(cm, cbar=False, cmap="Blues", annot=True, annot_kws={"size": 16}, fmt='g')
target_names = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sadness', "Surprise"]
print(classification_report(validation_generator.classes, y_pred, target_names=target_names))

