import cv2
import numpy as np

# Enable camera
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 420)

# Color detection parameters
""" lower_black = np.array([0], dtype=np.uint8)  # Lower grayscale value for black
upper_black = np.array([60], dtype=np.uint8)  # Upper grayscale value for black (dark gray)

lower_white = np.array([160], dtype=np.uint8)  # Lower grayscale value for white (including gray)
upper_white = np.array([255], dtype=np.uint8)  # Upper grayscale value for white """

lower_black = np.array([0, 0, 0], dtype=np.uint8)  # Lower BGR value for black
upper_black = np.array([60, 60, 60], dtype=np.uint8)  # Upper BGR value for black (dark gray)

lower_white = np.array([160, 160, 160], dtype=np.uint8)  # Lower BGR value for white (including gray)
upper_white = np.array([255, 255, 255], dtype=np.uint8)  # Upper BGR value for white

# import cascade file for facial recognition
# https://github.com/opencv/opencv/tree/master/data/haarcascades
catFaceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalcatface.xml")
catFaceCascadeExtended = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalcatface_extended.xml")

while True:
    success, img = cap.read()
    # imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    facesExt = catFaceCascade.detectMultiScale(img, scaleFactor=1.5, minNeighbors=2)
    facesExt2 = catFaceCascadeExtended.detectMultiScale(img, scaleFactor=1.5, minNeighbors=2)
    combined_faces = np.concatenate((facesExt, facesExt2), axis=0)

    for (x, y, w, h) in combined_faces:
        roi = img[y:y + h, x:x + w]  # Region of interest

         # Detect black color
        mask_black = cv2.inRange(roi, lower_black, upper_black)
        contours_black, _ = cv2.findContours(mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Check if the subject is black
        if len(contours_black) > 0:
            cv2.drawContours(img, contours_black, -1, (0, 0, 0), 2)
            cv2.putText(img, 'Tater', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        else:
            # Detect white color
            mask_white = cv2.inRange(roi, lower_white, upper_white)
            contours_white, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, contours_white, -1, (255, 255, 255), 2)
            cv2.putText(img, 'Pee', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(img, 'Cat', (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('cat_detect', img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyWindow('cat_detect')