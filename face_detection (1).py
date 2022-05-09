# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 18:28:38 2022

@author: Ahmed
"""


import cv2
import face_recognition

#loading the image to detect
image_to_detect = cv2.imread('./trump-modi.jpg')

#find the faces locations
all_face_locations = face_recognition.face_locations(image_to_detect,model="hog")

#print calculated number of faces
print("there are {} faces in this image" .format(len(all_face_locations)))

#looping through the face locations
for index, current_face_location in enumerate(all_face_locations):
    top_pos, right_pos,bottom_pos, left_pos  = current_face_location
    print("Found face {} at top: {}, right: {}, bottom: {}, left: {} "
          .format(index+1, top_pos,right_pos,bottom_pos,left_pos))
    current_face_image = image_to_detect[top_pos: bottom_pos, left_pos: right_pos]
    cv2.imshow("face N° "+str(index+1),current_face_image)
    