import cv2
import numpy as np
import os
from PIL import Image

labels = ["ChoiWoongJun","ChoiYeongHwan","HanJunHee", "KimJungMin","ParkSeungJae"]

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face-trainner.yml")#저장된 값 불러오기

cap = cv2.VideoCapture(0) #웹캠 실행

if cap.isOpened() == False:
    exit()

font = cv2.FONT_HERSHEY_SIMPLEX
while True:
    ret, img = cap.read() #현재 카메라에서 프레임마다 이미지 가져오기
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#흑백변환
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5,minNeighbors = 5)

    
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]#얼굴부분만 사진에서 가져오기
        id_,conf = recognizer.predict(roi_gray)#얼굴 유사도 확인 conf는 신뢰도
        print(id_,labels[id_],conf)
        
        if conf >=70:
            name = labels[id_] #이름리스트에서 이름 가져오기
            cv2.putText(img,name,(x,y),font,1,(0,0,255),2)
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

        else:
            name = "Outsider!"
            cv2.putText(img, name, (x,y),font,1,(0,0,255),2)
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow('Face Detect',img) #이미지 보여주기
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
