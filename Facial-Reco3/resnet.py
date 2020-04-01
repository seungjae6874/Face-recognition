from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
import numpy as np
import os
from PIL import Image

Face_Images = os.path.join(os.getcwd(),"Face_Images\ChoiWoongJun")

#mtcnn = MTCNN(image_size = <image_size>,margin=<margin>)

resnet = InceptionResnetV1(pretrained='vggface2').eval()
#img = cv2.imread(Face_Images)
img = Image.open("Face_Images\ChoiWoongJun")

#img_cropped = mtcnn(img, save_path = <"Face_images\ChoiWoonJun">)

img_embedding = resnet(img_cropped.unsqueeze(0))

resnet.classify = True
img_probs = resnet(img_cropped.unsqueeze(0))
