#Read a videp from web cam using openCV
#Face Detection in Video
#Click 20 Pictures of the
import cv2
import numpy as np
from PIL.ImageChops import offset

cam= cv2.VideoCapture(0)
fileName=input("Enter the name of the person : ")
dataset_path="./data/"
offset=20

model= cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
faceData=[]
skip=0
while True:
    success, img = cam.read()
    if not success:
        print("Failed to capture image")
    #store the gray image
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = model.detectMultiScale(img, 1.3, 5)
    faces = sorted(faces, key = lambda f : f[2]*f[3])
    if len(faces) > 0:
      f=faces[-1]
      x, y, w, h = f
      cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)

      cropped_face = img[y-offset:y+h+offset, x-offset:x+offset+w]
      if cropped_face is not None and cropped_face.size != 0:
        cropped_face=cv2.resize(cropped_face,(100,100))
      skip+=1
      if skip%10==0:
          if cropped_face is not None and cropped_face.size != 0:
           faceData.append(cropped_face)
           print("Saved so far " +str(len(faceData)))

    cv2.imshow("Image Window",img)
       #cv2.imshow("Cropped Window",cropped_face)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
#Writw the faceDaya on the dsik
faceData=np.asarray(faceData)
m= faceData.shape[0]
faceData=faceData.reshape((m,-1))
print(faceData.shape)
filePath=dataset_path+fileName+".npy"
np.save(filePath,faceData)
print("Data saved successfully"+filePath)
cam.release()
cv2.destroyAllWindows()
cam.release()
cv2.destroyAllWindows()
