import numpy as np
import cv2

# Load face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')

# Load face detector
cascadePath = 'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascadePath)

# Set font and labels
font = cv2.FONT_HERSHEY_TRIPLEX
names = ['0', '1', '2', '3', '4', '5']

# Open camera
cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)

# Set minimum face size
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

# Start loop
while True:
    # Read frame from camera
    ret, img = cam.read()

    # Convert frame to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH))
    )

    # Process each face
    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Recognize face
        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

        # Set label and confidence text
        if confidence < 100:
            label = names[id]
            confidence_text = "{0}%".format(round(100 - confidence))
        else:
            label = "unknown"
            confidence_text = "{0}%".format(round(confidence))

        # Draw label and confidence text
        cv2.putText(img, label, (x + 5, y + h - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, confidence_text, (x + 5, y + h + 25), font, 1, (255, 255, 0), 1)

    # Show image
    cv2.imshow('camera', img)

    # Check for exit key
    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break

# Cleanup
print("\n [INFO] Exiting program")
cam.release()
cv2.destroyAllWindows()