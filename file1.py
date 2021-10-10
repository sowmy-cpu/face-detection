# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 17:27:28 2021

@author: H.SOUMYA
"""
import cv2
import os
import numpy as np

# this module consists of the functions that we will use to recognize the face

# Given an image, this function returns rectangle for face detected with grayscale image
def faceDetection(test_img):
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    face_haar_cascade = cv2.CascadeClassifier(r'C:\\Users\\H.SOUMYA\\Documents\\facedetectionproject\\Haarcascade\\haarcascade.xml') #Lading the Haar cascade file
    
    faces = face_haar_cascade.detectMultiScale(gray_img)
    # detectMultiscale reurns rectangle
    return faces, gray_img

# give labels to training images. This function takes a directory path as input
def labels_for_training_images(directory):
    faces=[]
    faceID=[]
    
    for path,subdirnames,filenames in os.walk(directory):
        for filename in filenames:
            if filename.startswith('.'):
                print('Skipping the system file')
                continue
            id = os.path.basename(path)
            img_path = os.path.join(path, filename) #fetching image path
            print("image path: ", img_path)
            test_img = cv2.imread(img_path) # loads the image one by one
            if test_img is None:
                print('Image not loades properly')
                continue
            faces_rect, gray_img = faceDetection(test_img) 
            # calling the faceDetection function to return face location in each images
            if len(faces_rect)!=1:
                continue # Since we are assuming only one person is there in the image
                        
            (x,y,w,h) = faces_rect[0]
            roi_gray =gray_img[y:y+h,x:x+w] # cropping region of interest i.e. face location
            faces.append(roi_gray)
            faceID.append(int(id))
    return faces, faceID


# Below function trains the classifier using the training images
def train_classifier(faces, faceID):
    facerecognizer = cv2.face.LBPHFaceRecognizer_create()
    facerecognizer.train(faces,np.array(faceID))
    return facerecognizer

# Below function draws a bounding box around the face
def draw_rect(test_img,face):
    (x,y,w,h) = face
    cv2.rectangle(test_img, (x,y), (x+w, y+h), (255,0,0), thickness=5)
    
# Below function writes the name of the person in the image
def put_text(test_img, text, x,y):
    cv2.putText(test_img,text,(x,y), cv2.FONT_HERSHEY_DUPLEX,2,(255,0,0),4)