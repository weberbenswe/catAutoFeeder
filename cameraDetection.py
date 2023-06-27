import cv2
import numpy as np

# Enable camera
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 420)

# Color detection parameters
lower_black = 0
upper_black = 64

# Object motion parameters
previous_frame = None
motion_threshold = 12000 # more movement required

# Object size parameters
previous_object_size = 0
size_change_threshold = 5 # object needs to be growing faster

# import cascade file for facial recognition
# https://github.com/opencv/opencv/tree/master/data/haarcascades
# catFaceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalcatface.xml")
# catFaceCascadeExtended = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalcatface_extended.xml")

while True:
    success, img = cap.read()
    imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    mask = cv2.inRange(imgGrey, lower_black, upper_black)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_contour_area = 75 # lower == smaller detected objects
    min_perimeter_ratio = 4 # higher number restricts more roundness


    for contour in contours:
        confidence = 0
        result = ''

        # Must meet min size threshold
        if cv2.contourArea(contour) > min_contour_area:
            confidence += 1

            # Object is black
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
            x, y, w, h = cv2.boundingRect(approx)
            confidence+=1
            result += 'COLOR '

            # Object shape detection *
            contour_area = cv2.contourArea(contour)
            contour_perimeter = perimeter
            perimeter_ratio = contour_perimeter / (2 * np.sqrt(contour_area / np.pi))

            if perimeter_ratio > min_perimeter_ratio:
                # print('ratio', perimeter_ratio)
                confidence += 1
                result += 'SHAPE '

            # Object motion detection * 
            current_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if previous_frame is not None:
                
                frame_diff = cv2.absdiff(current_frame, previous_frame)
                motion = np.sum(frame_diff > 30)

                if motion > motion_threshold:
                    # print('motion', motion)
                    confidence+=1
                    result+='MOVING '
                    
                # Object size change detection * 
                current_object_size = w * h
                if previous_object_size != 0:
                    size_change = abs(current_object_size - previous_object_size) / previous_object_size
                    if size_change > size_change_threshold:
                        print('Size', size_change)
                        confidence+=1
                        result+='GROWING '
                    
                previous_object_size = current_object_size
            previous_frame = current_frame.copy()

        if confidence > 0:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(img, str(result), (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('cat_detect', img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyWindow('cat_detect')