import cv2
import os.path
file='haarcascade_frontalface_default.xml'
face=cv2.CascadeClassifier(file)
cap=cv2.VideoCapture(0)

folder_name="dataset"
sub_folder="photos"

# path=os.path.join(folder_name,sub_folder)
# if not os.path.isdir(path):
#     os.mkdir(path)

path='D:\Python work\AI-work\Face-detection-and-saving-Detectedfaces-images-locally\dataset'

count=1
while (count<30):
    print(count)
    _,img=cap.read()
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces=face.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        only_face=gray[y:y+h,x:x+w]
        cv2.imwrite("%s/%s.jpg" %(path,count),only_face)
        count+=1
    cv2.imshow("frame",img)
    key=cv2.waitKey(10)
    if key==27:
        break
print("dataset created")
cap.release()
cv2.destroyAllWindows()