# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 18:28:38 2022

@author: Ahmed
"""

import face_recognition
import cv2

#capture video from default camera
webcam_video_stream = cv2.VideoCapture(0 + cv2.CAP_DSHOW)

#intialize the array variable to hold all face locations 
all_face_locations = []

#loop through every frame in the video
while True:
    #get the current frame from the video stream as an image
    ret, current_frame = webcam_video_stream.read()
    current_frame_small = cv2.resize(current_frame,(0,0), fx=0.25, fy=0.25)
    all_face_locations = face_recognition.face_locations(current_frame_small,number_of_times_to_upsample=2,model="hog")
    #looping through the face locations
    for index, current_face_location in enumerate(all_face_locations):
        top_pos, right_pos,bottom_pos, left_pos  = current_face_location
        top_pos = top_pos*4
        right_pos = right_pos * 4
        bottom_pos = bottom_pos * 4
        left_pos = left_pos * 4
        print("Found face {} at top: {}, right: {}, bottom: {}, left: {} "
              .format(index+1, top_pos,right_pos,bottom_pos,left_pos))
        current_face_image = current_frame_small[top_pos: bottom_pos, left_pos: right_pos]
        #cv2.imshow("face N° "+str(index+1),current_face_image)
        cv2.rectangle(current_frame, (left_pos, top_pos), (right_pos, bottom_pos), (0,0,255), 2)
    #showing the current face with rectangle drawn
    cv2.imshow("webcam video", current_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
webcam_video_stream.release()
cv2.destroyAllWindows()