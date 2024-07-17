import cv2

GREEN = (0,255,0)

faceDetector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

imatge = cv2.imread("resources/nba2.webp")

imatgeGris = cv2.cvtColor(imatge, cv2.COLOR_BGR2GRAY)

result = faceDetector.detectMultiScale(imatgeGris)

# [ [ CoordX, CoordY, Amplada, Altura ] ]
for rectangle in result:
    #dins del bucle
    (coordX, coordY, amplada, altura) = rectangle
    print("Cara trobada a lees coordenades:", coordX, coordY, amplada, altura)
    cv2.rectangle(imatge, (coordX, coordY), (coordX+amplada, coordY+altura), GREEN, 2)
#fora del bucle

cv2.imshow("Imatge sense detectar cares", imatge)

cv2.waitKey()


print("Programa Acabat!")