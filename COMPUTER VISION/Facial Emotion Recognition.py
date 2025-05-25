#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/MaharshSuryawala/Face-Detection-and-Facial-Expression-Recognition/blob/master/FaceDetectionAndExpressionRecognition_ML.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[ ]:


get_ipython().system('pip install -U -q PyDrive')
import os

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

#import google.colab import drive
#drive.mount('/gdrive')

# 1. Authenticate and create the PyDrive client.
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

# choose a local (colab) directory to store the data.
local_download_path = os.path.expanduser('~/ML')

try:
  os.makedirs(local_download_path)
except: pass


# In[ ]:


# 2. Auto-iterate using the query syntax
#    https://developers.google.com/drive/v2/web/search-parameters
file_list = drive.ListFile(
    {'q': "'12efh4IZ9r54HgrnPULW0qgFk2MjIDR5A' in parents"}).GetList()

for f in file_list:
  # 3. Create & download by id.
  print('title: %s, id: %s' % (f['title'], f['id']))
  fname = os.path.join(local_download_path, f['title'])
  print('downloading to {}'.format(fname))
  f_ = drive.CreateFile({'id': f['id']})
  f_.GetContentFile(fname)


# 

# In[ ]:


import numpy as np
from matplotlib import pyplot as plt
#py.init_notebook_mode(connected=True)
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import numpy as np
import math
from scipy.stats import norm
import time
import pandas as pd
import statsmodels.api as sm
from keras.layers import Dense, Dropout, LSTM, Embedding
from keras.models import Sequential
from matplotlib import pyplot as plt
import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode(connected=True)
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection
from keras.utils.np_utils import to_categorical
get_ipython().run_line_magic('matplotlib', 'inline')

#from PIL import Image
#from resizeimage import resizeimage


# In[ ]:


get_ipython().run_line_magic('cd', '/root/ML')

data = pd.read_csv('Train_Roll_Number.txt', sep=" ", header=None)
data1 = pd.read_csv('Train_RGB_Sketch.txt', sep=" ", header=None)
data2 = pd.read_csv('Train_Gender.txt', sep=" ", header=None)
data3 = pd.read_csv('Train_Expression.txt', sep=" ", header=None)

data.columns = ["filename", "subject"]
data1.columns = ["filename", "sketch"]
data2.columns = ["filename", "gender"]
data3.columns = ["filename", "expression"]

data['X'] = None
X = None

set_type = set()
set_shape = set()

for index, row in data.iterrows():
    # print(row["filename"], row["X"], row["subject"])
    
    im = cv2.imread(row["filename"])
    
    if row["filename"] == "1641010_Male_Neutral_Sketch.png":
        NoneType = type(im)
    
        break

count_none = 0
  
width = 420
height = 240

for index, row in data.iterrows():
    # print(row["filename"], row["X"], row["subject"])
    
    im = cv2.imread(row["filename"])
    
    if type(im) != None:
        
        im = cv2.resize(im, (height, width), interpolation = cv2.INTER_CUBIC)
        print(row["filename"])
        vec = np.array(np.reshape(im, (-1, 1)))
        
        data['X'][index] = vec
        
        """
        if ((data1['filename'] == row['filename']) & (data1['sketch'] == 0)).any():
            if index == 0:
                X = vec
                y_subject = np.array(data['subject'][index])

            else:
                X = np.concatenate((X, vec), axis=1)
                y_subject = np.concatenate((y_subject, np.array(data['subject'][index])), axis=1)
        """
        
    if type(im) == None:
        count_none += 1
        print(row["filename"])
    
print(count_none)

# Let's print the first ten values for the signal vector
print(data.head())


# In[ ]:


data = pd.merge(data, data1, on=['filename'])
data = pd.merge(data, data2, on=['filename'])
data = pd.merge(data, data3, on=['filename'])


# In[ ]:


# Deleting None rows
print(data.shape)
data = data.dropna()
print(data.shape)
data


# In[ ]:


data_nosketch = data[data['sketch'] == 0]
data_nosketch


# In[ ]:


def return_dataset(data, class_name):
    for index, row in data.iterrows():
        
        vec = data['X'][index]
      
        if index == 0:
            X = vec
            y = np.reshape(data[class_name][index], (1, 1))

        else:
            X = np.concatenate((X, vec), axis=1)
            y = np.concatenate((y, np.reshape(data[class_name][index], (1, 1))), axis=0)
            
    return X, y


# In[ ]:


X, y = return_dataset(data, 'subject')
X = X.T

# print(X.shape)

#train_x, test_x, train_y, test_y = model_selection.train_test_split(X, y, test_size = 0.1, random_state = 0)

#X = X.astype(np.float32)
#X = X.T
#print(type(X))

X_train_DT, X_test_DT, y_train_DT, y_test_DT = model_selection.train_test_split(X, data['subject'], test_size = 0.1, random_state = 0)
X_train_KNN, X_test_KNN, y_train_KNN, y_test_KNN = model_selection.train_test_split(X, data['subject'], test_size = 0.1, random_state = 0)
X_train_SVM, X_test_SVM, y_train_SVM, y_test_SVM = model_selection.train_test_split(X, data['subject'], test_size = 0.1, random_state = 0)
X_train_RF, X_test_RF, y_train_RF, y_test_RF = model_selection.train_test_split(X, data['subject'], test_size = 0.1, random_state = 0)
X_train_NB, X_test_NB, y_train_NB, y_test_NB = model_selection.train_test_split(X, data['subject'], test_size = 0.1, random_state = 0)


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
classifier_DT = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier_DT.fit(X_train_DT, y_train_DT)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
classifier_KNN = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier_KNN.fit(X_train_KNN, y_train_KNN)


# In[ ]:


from sklearn.svm import SVC
classifier_SVM = SVC(kernel = 'linear', random_state = 0)
classifier_SVM.fit(X_train_SVM, y_train_SVM)


# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_NB = sc.fit_transform(X_train_NB)
X_test_NB = sc.transform(X_test_NB)

from sklearn.naive_bayes import GaussianNB
classifier_NB = GaussianNB()
classifier_NB.fit(X_train_NB, y_train_NB)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
classifier_RF = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier_RF.fit(X_train_RF, y_train_RF)


# In[ ]:


y_pred_DT = classifier.predict(X_test_DT)
y_test_DT = y_test_DT.tolist() 

y_pred_SVM = classifier.predict(X_test_SVM)
y_test_SVM = y_test_SVM.tolist() 

y_pred_RF = classifier.predict(X_test_RF)
y_test_RF = y_test_RF.tolist() 

y_pred_NB = classifier.predict(X_test_NB)
y_test_NB = y_test_NB.tolist() 

y_pred_KNN = classifier.predict(X_test_KNN)
y_test_KNN = y_test_KNN.tolist() 


# In[ ]:


y_test_np_DT = np.asarray(y_test_DT)
y_test_np_SVM = np.asarray(y_test_SVM)
y_test_np_RF = np.asarray(y_test_RF)
y_test_np_KNN = np.asarray(y_test_KNN)
y_test_np_NB = np.asarray(y_test_NB)


# In[ ]:


# Accuracy
##############################################
disc_NB = y_test_np_NB - y_pred_NB

count = 0
for i in disc_NB:
  if i == 0:
    count += 1

accuracy_NB = ( (100 * count) / len(y_pred_NB))
##############################################
disc_RF = y_test_np_RF - y_pred_RF

count = 0
for i in disc_RF:
  if i == 0:
    count += 1

accuracy_RF = ( (100 * count) / len(y_pred_RF))
##############################################
disc_KNN = y_test_np_KNN - y_pred_KNN

count = 0
for i in disc_KNN:
  if i == 0:
    count += 1

accuracy_KNN = ( (100 * count) / len(y_pred_KNN))
##############################################
disc_SVM = y_test_np_SVM - y_pred_SVM

count = 0
for i in disc_SVM:
  if i == 0:
    count += 1

accuracy_SVM = ( (100 * count) / len(y_pred_SVM))
##############################################
disc_DT = y_test_np_DT - y_pred_DT

count = 0
for i in disc_DT:
  if i == 0:
    count += 1

accuracy_DT = ( (100 * count) / len(y_pred_DT))
##############################################



print(accuracy_NB)
print(accuracy_RF)
print(accuracy_DT)
print(accuracy_KNN)
print(accuracy_SVM)


# In[ ]:




