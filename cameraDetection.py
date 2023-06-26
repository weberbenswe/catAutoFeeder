import cv2

# Enable camera
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 420)

# import cascade file for facial recognition
catFaceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalcatface.xml")
catFaceCascadeExtended = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalcatface_extended.xml")

while True:
    success, img = cap.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    facesExt = catFaceCascadeExtended.detectMultiScale(imgGray, scaleFactor=1.05, minNeighbors=6, minSize=(30,30))
    for (x, y, w, h) in facesExt:
        print()
        imgGray = cv2.rectangle(imgGray, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(imgGray, 'Cat', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('cat_detect', imgGray)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyWindow('cat_detect')

# TODO: learn difference between cats in greyscale, adjust detection, look into ordering parts