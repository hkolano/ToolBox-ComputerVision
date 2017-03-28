""" Experiment with face detection and image filtering using OpenCV """

import cv2
import numpy as np

# get the face detection file
face_cascade = cv2.CascadeClassifier('/home/kyle/softdes/ToolBox-ComputerVision/haarcascade_frontalface_alt.xml')

# matrix for blurring
kernel = np.ones((21, 21), 'uint8')

# initialize camera
camera = cv2.VideoCapture(0)

while True:
    # Capture frame
    ret, frame = camera.read()

    # Detect faces
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.2, minSize=(50,50))

    # blur and draw on the faces
    for (x, y, w, h) in faces:
        frame[y:y+h, x:x+w, :] = cv2.dilate(frame[y:y+h, x:x+w, :], kernel)
        cv2.ellipse(frame,(x+int(w/2), y+int(h/3)), (int(w/3),int(h/2)), 0, 60, 120, (0,0,0), 4)
        cv2.circle(frame, (x+int(w/3), y+int(h/3)), 6, (255,255,255), 8)
        cv2.circle(frame, (x + int(2*w / 3), y + int(h / 3)), 6, (255, 255, 255), 8)
        # cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255))

    # Display frame
    cv2.imshow('Webcam', frame)

    # Watch for exit key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
