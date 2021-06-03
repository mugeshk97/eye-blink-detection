import cv2
import numpy as np
import dlib
from math import hypot


detector = dlib.get_frontal_face_detector() # face detector loaded
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") # face landmark model loaded
# function to calculate the midpoint of two points x and y
def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

# function to to calculate the blinking_ratio
def get_blinking_ratio(eye_points, facial_landmarks):
    left = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y) #left point of eye
    right = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y) #right_point of eye
    top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2])) # center_top
    bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4])) # center_bottom
    # draw the line between points
    hor_line = cv2.line(frame, left, right, (0, 255, 0), 2)
    ver_line = cv2.line(frame, top, bottom, (0, 255, 0), 2)
    hor_length = hypot((left[0] - right[0]), (left[1] - right[1]))# calculate the distance between left and right
    ver_length = hypot((top[0] - bottom[0]), (top[1] - bottom[1]))# calculate the distance between top and bottom
    # calculate the ratio
    ratio = hor_length / ver_length
    return ratio

frame = cv2.cv2.imread('img2.jpeg') # reading the image file
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert it into grey scale

faces = detector(gray) # detecting the faces in the image
for face in faces: # looping through all the faces
    
    x, y = face.left(), face.top() 
    x1, y1 = face.right(), face.bottom()
    cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2) #bounding box around the face

    landmarks = predictor(gray, face) # predicting the landmarks in the face

    left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks) # calculate the left_eye_ratio
    right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks) #calculate the right_eye_ratio
    blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2 # calculate the overall ratio
    if blinking_ratio > 5.7:
        cv2.putText(frame, "EYES CLOSED", (50, 150), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 3) # writing the output in the image
    else:
        cv2.putText(frame, "EYES OPENED", (50, 150), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)
cv2.imshow("Frame", frame)# display the output
cv2.waitKey(0) # assign ESC key to quit
cv2.destroyAllWindows() #close all the output windows
