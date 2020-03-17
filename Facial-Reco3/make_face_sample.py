import cv2
import numpy as np
#얼굴 인식용 xml파일
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#전체 사진에서 얼굴 부위만 잘라 리턴하는 함수
def face_extractor(img):

    #흑백 처리
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #얼굴 찾기 face_classifier의 detect를 사용한다.
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    #찾는 얼굴이 없으면 None으로 리턴한다.
    if faces is():
        return None

    #사진에서 얼굴을 찾았다면 x,y,w,h 값을 얻어 처리 x,y는 왼쪽 아래 점.
    for(x,y,w,h) in faces:

        #얼굴사이즈 만큼 crop하며 잘라넣는다
        #만약 얼굴이 2개 이상 감지 되었을 시에 가장 마지막 얼굴만 남긴다.
        cropped_face = img[y:y+h, x:x+w]
    #crop 한 얼굴사진을 리턴
    return cropped_face


#카메라 실행
cap = cv2.VideoCapture(0)

#저장할 이미지 카운트 변수

slist = [1]
name = 'ChoiWoongJun'
for i in slist:
    count = 0    
    while True:

    #카메라로 부터 사진 1장을 찍어 얻기
        ret, frame = cap.read()

    #얼굴 감지 함수를 사용해서 크롭한 얼굴사진 가져오기
        if face_extractor(frame) is not None:
            count+=1

        #얼굴 이미지 크기를 200*200사이즈로 리사이즈
            face = cv2.resize(face_extractor(frame),(200,200))
        #조정한 이미지를 흑백사진으로 변환
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        #faces폴더에 user#로 .jpg로 저장
            file_name_path = 'Face_Images/%s/'%name+str(count)+'.jpg'
            cv2.imwrite(file_name_path,face)

        #화면에 얼굴과 현재 저장개수 표시해준다
            cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            cv2.imshow('Face Cropper',face)
        else:
            print("Face not Found")
            pass

        if cv2.waitKey(1)==13 or count==100:
            break

cap.release()
cv2.destroyAllWindows()
print('Colleting Samples Complete!!!')
