import cv2
import numpy as np

# Enable camera
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 420)

# Color detection parameters
lower_black = np.array([0, 0, 0])  # Lower range of black color (in HSV)
upper_black = np.array([180, 255, 30])  # Upper range of black color (in HSV)
lower_white = np.array([0, 0, 200])  # Lower range of white color (in HSV)
upper_white = np.array([180, 30, 255])  # Upper range of white color (in HSV)

# import cascade file for facial recognition
# https://github.com/opencv/opencv/tree/master/data/haarcascades
catFaceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalcatface.xml")
catFaceCascadeExtended = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalcatface_extended.xml")

while True:
    success, img = cap.read()
    imgHSV  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    facesExt = catFaceCascadeExtended.detectMultiScale(imgHSV, scaleFactor=1.05, minNeighbors=6, minSize=(30,30))
    
    for (x, y, w, h) in facesExt:
        roi = imgHSV[y:y + h, x:x + w]  # Region of interest

         # Detect black color
        mask_black = cv2.inRange(roi, lower_black, upper_black)
        contours_black, _ = cv2.findContours(mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Check if the subject is black
        if len(contours_black) > 0:
            cv2.drawContours(imgHSV, contours_black, -1, (0, 0, 0), 2)
            cv2.putText(imgHSV, 'Tater', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        else:
            # Detect white color
            mask_white = cv2.inRange(roi, lower_white, upper_white)
            contours_white, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(imgHSV, contours_white, -1, (255, 255, 255), 2)
            cv2.putText(imgHSV, 'Pee', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.rectangle(imgHSV, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(imgHSV, 'Cat', (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('cat_detect', imgHSV)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyWindow('cat_detect')