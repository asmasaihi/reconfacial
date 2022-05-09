# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 18:28:38 2022

@author: Ahmed
"""

import face_recognition
import cv2
import smtplib

#capture video from default camera
#0 + cv2.CAP_DSHOW
webcam_video_stream = cv2.VideoCapture("./sampels/123.avi")

#spécifier les encodings des visages

wangchong_image = face_recognition.load_image_file('./sampels/images/wangchong.jpg')
wangchong_face_encodings = face_recognition.face_encodings(wangchong_image)[0]

ahmed_image = face_recognition.load_image_file('./sampels/images/ahmed.jpg')
ahmed_face_encodings = face_recognition.face_encodings(ahmed_image)[0]

#stocker les encodings dans un tableau et les noms dans un autre dans le mème ordre
known_face_encoding = [wangchong_face_encodings, ahmed_face_encodings]
known_face_names = ["wangchong", "ahmed"]

#intialize the array variable to hold all face locations, encodings and names in the frame
all_face_locations = []
all_face_encodings = []
all_face_names = []
while True:
    #get the current frame from the video stream as an image
    ret, current_frame = webcam_video_stream.read()
    current_frame_small = cv2.resize(current_frame,(0,0), fx=0.25, fy=0.25)
    all_face_locations = face_recognition.face_locations(current_frame_small,number_of_times_to_upsample=2,model="cnn")
    
    all_face_encodings = face_recognition.face_encodings(current_frame_small, all_face_locations)
    #looping through the face locations
    for current_face_location, current_face_encoding in zip(all_face_locations, all_face_encodings):
        top_pos, right_pos,bottom_pos, left_pos  = current_face_location
        top_pos = top_pos*4
        right_pos = right_pos * 4
        bottom_pos = bottom_pos * 4
        left_pos = left_pos * 4
       
        current_face_image = current_frame_small[top_pos: bottom_pos, left_pos: right_pos]
        #cv2.imshow("face N° "+str(index+1),current_face_image)
        cv2.rectangle(current_frame, (left_pos, top_pos), (right_pos, bottom_pos), (0,0,255), 2)
        #find all the matches and get the list of all matches
        all_matches = face_recognition.compare_faces(known_face_encoding, current_face_encoding)
        #string to hold the label
        name_of_person = 'Unknown face'
        #chek if all_matches have at least one item
        #if yes, get the index number of face that is located in the first index of all_matches
        if True in all_matches:
            first_match_index = all_matches.index(True)
            name_of_person = known_face_names[first_match_index]
        #else:
            
        #draw rectangle around the face
        cv2.rectangle(current_frame_small, (left_pos, top_pos), (right_pos, bottom_pos), (255,0,0), 2)
        
        #display the name as text in the image
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(current_frame, name_of_person, (left_pos, bottom_pos),  font, 0.5, (255,255,255), 1)
        
        
        
    #showing the current face with rectangle drawn
    cv2.imshow("webcam video", current_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
webcam_video_stream.release()
cv2.destroyAllWindows()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# =============================================================================
#     for index, current_face_location in enumerate(all_face_locations):
#         top_pos, right_pos,bottom_pos, left_pos  = current_face_location
#         top_pos = top_pos*4
#         right_pos = right_pos * 4
#         bottom_pos = bottom_pos * 4
#         left_pos = left_pos * 4
#         print("Found face {} at top: {}, right: {}, bottom: {}, left: {} "
#               .format(index+1, top_pos,right_pos,bottom_pos,left_pos))
#         current_face_image = current_frame_small[top_pos: bottom_pos, left_pos: right_pos]
#         #cv2.imshow("face N° "+str(index+1),current_face_image)
#         cv2.rectangle(current_frame, (left_pos, top_pos), (right_pos, bottom_pos), (0,0,255), 2)
#     #showing the current face with rectangle drawn
#     cv2.imshow("webcam video", current_frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# webcam_video_stream.release()
# cv2.destroyAllWindows()
# =============================================================================
