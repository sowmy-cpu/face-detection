# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 17:49:49 2021

@author: H.SOUMYA
"""

import cv2
import file1 as fr

test_img=cv2.imread(r'C:/Users/H.SOUMYA/Documents/facedetectionproject/Test_img/test_img.jpg')
print(test_img)
faces_detected,gray_img = fr.faceDetection(test_img)
print('face detected:', faces_detected)

# comment these lines when you are running the code from the second time
faces, faceID = fr.labels_for_training_images(r'C:/Users/H.SOUMYA/Documents/facedetectionproject/Training_imgs')
face_recognizer = fr.train_classifier(faces, faceID)
face_recognizer.write(r'C:/Users/H.SOUMYA/Documents/facedetectionproject/trainingdata.yml')

# uncomment these lines while running the code from second time onwards
#face_recognizer=cv2.face.LBPHFaceRecognizer_create()
#face_recognizer.read(r'C:/Users/H.SOUMYA/Documents/facedetectionproject/trainingdata.yml')


name = {0:'Sowmya'} # creating dictionary containing names for label

for face in faces_detected:
    (x,y,w,h)=face
    roi_gray=gray_img[y:y+h,x:x+w]
    label, confidence = face_recognizer.predict(roi_gray) #predicts the label of the image
    
    fr.draw_rect(test_img, face)
    predicted_name = name[label]
    fr.put_text(test_img, predicted_name, x, y)
    print('Confidence:', confidence)
    print('label', label)
    resized_img = cv2.resize(test_img,(500,700))
    cv2.imshow('face detection',resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows
