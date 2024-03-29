####################################################
# Modified by Nazmi Asri                           #
# Original code: http://thecodacus.com/            #
# All right reserved to the respective owner       #
####################################################

# Import OpenCV2 for image processing
import cv2

# Import numpy for matrices calculations
import numpy as np
import boto3

import os
import time
from drowsiness_detect import sleep

s3=boto3.client('s3',aws_access_key_id='AKIAIJT2QJO6GXJ6HXIQ', aws_secret_access_key='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') 

def reck():
    def assure_path_exists(path):
        dir = os.path.dirname(path)
        if not os.path.exists(dir):
            os.makedirs(dir)

    # Create Local Binary Patterns Histograms for face recognization
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    assure_path_exists("trainer/")

    # Load the trained mode
    recognizer.read('trainer/trainer.yml')

    # Load prebuilt model for Frontal Face
    cascadePath = "haarcascades/haarcascade_frontalface_default.xml"

    # Create classifier from prebuilt model
    faceCascade = cv2.CascadeClassifier(cascadePath);

    # Set the font style
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Initialize and start the video frame capture
    cam = cv2.VideoCapture(0)
   
    # Loop
    while True:
        # Read the video frame
        ret, im =cam.read()

        # Convert the captured frame into grayscale
        gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

        # Get all face from the video frame
        faces = faceCascade.detectMultiScale(gray, 1.2,5)

        # For each face in faces
        for(x,y,w,h) in faces:

            # Create rectangle around the face
            cv2.rectangle(im, (x-20,y-20), (x+w+20,y+h+20), (0,255,0), 4)

            # Recognize the face belongs to which ID
            Id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

            # Check the ID if exist 
            
            if(Id == 1):
                n="{0:.2f}".format(round(100 - confidence, 2))
                print(n)
                if float(n)<30:
                    print('ok')
                    cv2.imwrite("img_name.jpg", im)
                    s3.upload_file('img_name.jpg', 'sleep1ing', 'image.jpg')
                    cv2.destroyAllWindows()
                    sleep()
                

            
            # Put text describe who is in the picture
            cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), (0,255,0), -1)
            cv2.putText(im, (x,y-40), font, 1, (255,255,255), 3)

        # Display the video frame with the bounded rectangle
        cv2.imshow('im',im) 
    cv2.destroyAllWindows()
    #     # If 'q' is pressed, close program
    #     if cv2.waitKey(10) & 0xFF == ord('q'):
    #         break

    # # Stop the camera
    # cam.release()

    # # Close all windows
    # cv2.destroyAllWindows()
reck()
