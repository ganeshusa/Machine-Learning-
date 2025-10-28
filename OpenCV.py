import cv2
from cv2 import imshow
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('School.png')
new_ig = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#print(img)

#Show this image
plt.imshow(new_ig)
plt.axis('off')

#face dection in image using OpenCV
model = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
faces = model.detectMultiScale(new_ig, 1.3, 1)

for f in faces:
        x, y, w, h = f
        cv2.rectangle(new_ig, (x, y), (x + w, y + h), (0,255, 0), 4)
plt.imshow(new_ig)
plt.axis('off')
plt.show()


# Face Dection in Live Video
cam= cv2.VideoCapture(0)
model= cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
while True:
    success, img = cam.read()
    if not success:
        print("Failed to capture image")
    faces = model.detectMultiScale(img, 1.3, 5)
    for f in faces:
        x, y, w, h = f
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)

    cv2.imshow("Image Window",img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
