import cv2

ESC = 27
WINDOW_NAME = 'Very Professional Webcam Face Detector'

def showWebcam(faceDetector):
    cam = cv2.VideoCapture(1)
    while True:
        retVal, frame = cam.read()
        showImage(detectFace(frame, faceDetector))
        if cv2.waitKey(30) == ESC: 
            break
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            break
    cv2.destroyAllWindows()

def detectFace(image, faceDetector):
    greyscaledImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    result = faceDetector.detectMultiScale(greyscaledImage)
    for face in result:
        (x, y, w, h) = face
        cv2.rectangle(image, (x, y), (x+w, y+h), (0,255,0), 2)
    return image

def showImage(image):
    cv2.imshow(WINDOW_NAME, image)

def main():
    faceDetector = cv2.CascadeClassifier('resources/haarcascade_frontalface_default.xml')
    showWebcam(faceDetector)

if __name__ == '__main__':
    main()