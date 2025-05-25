#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
import cv2
import imutils
from tensorflow.keras.preprocessing.image import img_to_array
from imutils.video import VideoStream
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# In[2]:


prototxtPath=os.path.sep.join([r'C:\Users\Chaitanya\Documents\MachineLearning_scripts\Face Mask Detection\Face Detector','deploy.prototxt'])
weightsPath=os.path.sep.join([r'C:\Users\Chaitanya\Documents\MachineLearning_scripts\Face Mask Detection\Face Detector','res10_300x300_ssd_iter_140000.caffemodel'])


# In[3]:


faceNN=cv2.dnn.readNet(prototxtPath,weightsPath)


# In[4]:


maskNN=load_model(r'C:\Users\Chaitanya\Documents\MachineLearning_scripts\Trained Models\maskdetect_mobilenet_v2_predictor.model')


# In[5]:


def detect_and_predict_mask(frame,faceNN,maskNN):

    (h,w)=frame.shape[:2]
    blob=cv2.dnn.blobFromImage(frame,1.0,(300,300),(104.0,177.0,123.0))
    
    faceNN.setInput(blob)
    detections=faceNN.forward()
    
    faces=[]
    locs=[]
    preds=[]
    
    for i in range(0,detections.shape[2]):
        confidence=detections[0,0,i,2]
        
        if confidence>0.5:
            box=detections[0,0,i,3:7]*np.array([w,h,w,h])
            (startX,startY,endX,endY)=box.astype('int')
            
            (startX,startY)=(max(0,startX),max(0,startY))
            (endX,endY)=(min(w-1,endX), min(h-1,endY))
            
           
            face=frame[startY:endY, startX:endX]
            face=cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
            face=cv2.resize(face,(224,224))
            face=img_to_array(face)
            face=preprocess_input(face)
            
            faces.append(face)
            locs.append((startX,startY,endX,endY))
            
        
        if len(faces)>0:
            faces=np.array(faces,dtype='float32')
            preds=maskNN.predict(faces,batch_size=12)
        
        return (locs,preds)


# In[6]:


vs=VideoStream(src=0).start()

while True:
    frame=vs.read()
    frame=imutils.resize(frame,width=600)
    
    (locs,preds)=detect_and_predict_mask(frame,faceNN,maskNN)
    
    for (box,pred) in zip(locs,preds):
        (startX,startY,endX,endY)=box
        (mask,withoutMask)=pred
        
        label='Mask' if mask>withoutMask else 'No Mask'
        color=(0,255,0) if label=='Mask' else (0,0,255)
        
        label="{}: {:.2f}%".format(label,max(mask,withoutMask)*100)
        
        cv2.putText(frame,label,(startX,startY-10),cv2.FONT_HERSHEY_SIMPLEX,0.45,color,2)        
        cv2.rectangle(frame,(startX,startY),(endX,endY),color,2)
        
    cv2.imshow('Video',frame)
    k = cv2.waitKey(1)   
    if(k == ord('q')):
        break       
        
cv2.destroyAllWindows()
vs.stop()


# In[ ]:





# In[ ]:





# In[ ]:




