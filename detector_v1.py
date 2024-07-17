import cv2

faceDetector = cv2.CascadeClassifier('resources/haarcascade_frontalface_default.xml')

inputImage = cv2.imread('resources/nba2.webp')

greyscaledInputImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)

result = faceDetector.detectMultiScale(greyscaledInputImage)

for face in result:
    (x, y, w, h) = face
    cv2.rectangle(inputImage, (x, y), (x+w, y+h), (0,255,0), 2)

cv2.imshow('Hello world!!', inputImage)
cv2.waitKey()