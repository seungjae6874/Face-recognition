import cv2
import numpy as np
import os
import random
from PIL import Image

#import augmenters from imgaug

#순서는 폴더의 경로에 따라서 1번 이미지를 읽어서 augmentation 실행
Face_Images = os.path.join(os.getcwd(), "Face_Images") #이미지파일 경로 지정
#print(Face_Images)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')



#augmentation
def random_crop(image, row_crop_pix = 25):
     rows, cols, _ = image.shape
     cols_crop_pix = round((row_crop_pix *cols)/rows)
     rows_crop_pix = int(np.floor(2*random.uniform(0,1)*row_crop_pix))
     cols_crop_pix = int(np.floor(2*random.uniform(0,1)*cols_crop_pix).astype(int))
     return image[row_crop_pix:rows-row_crop_pix, cols_crop_pix:cols - cols_crop_pix]

def rotate(image, angle = 90, scale = 1.0):
     w = image.shape[1]
     h = image.shape[0]
     #rotate matrix
     M = cv2.getRotationMatrix2D((w/2,h/2),angle, scale)
     #rotate
     image = cv2.warpAffine(image,M,(w,h))
     return image

def rotate2(image, angle = 45, scale = 1.0):
     w = image.shape[1]
     h = image.shape[0]
     #rotate matrix
     M = cv2.getRotationMatrix2D((w/2,h/2),angle, scale)
     #rotate
     image = cv2.warpAffine(image,M,(w,h))
     return image


def flip(image, vflip = False, hflip = False):
    if hflip or vflip:
        if hflip and vflip:
              c = -1
        else:
            c = 0 if vflip else 1
        image = cv2.flip(image,flipCode = c)
            
    return image

def bluring(image):
     kernel = np.ones((7,7),np.float32)/49
     blur = cv2.filter2D(image,-1,kernel)
     return blur
     


Face_ID = -1
pev_person_name = ""
name = ["ChoiWoongJun","ChoiYeongHwan","HanJunHee", "KimJungMin","ParkSeungJae"]
#차례로 이름 폴더마다 1장을 읽자
i = 0
for root,dirs,files in os.walk(Face_Images):
    count=600
    for file in files:
         i=0
         if i == 100:
              break
         if file.endswith("jpeg") or file.endswith("jpg") or file.endswith("png"):
            path = os.path.join(root,file)
            path2 = os.path.join(root)
            person_name = os.path.basename(root)
            print(path2, person_name)
            image = cv2.imread(path)
            aug1 = rotate(image)
            aug2 = flip(image)
            #count+=1
            #file_name_path = 'Face_Images/%s/'%person_name+str(count)+'.jpg'
            #cv2.imwrite(file_name_path,aug1)
            #count+=1
            #file_name_path2 = 'Face_Images/%s/'%person_name+str(count)+'.jpg'
            #cv2.imwrite(file_name_path2,aug2)
            count+=1
            #aug3 = bluring(image)
            #file_name_path3 = 'Face_Images/%s/'%person_name+str(count)+'.jpg'
            #cv2.imwrite(file_name_path3,aug3)
            aug4 = random_crop(image)
            file_name_path4 = 'Face_Images/%s/'%person_name+str(count)+'.jpg'
            cv2.imwrite(file_name_path4,aug4)
            i+=1
            
            
print("All Augmentation Done!")



            
