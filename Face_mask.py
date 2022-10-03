#   Welcome to my Face_mask
#   โหลด ไฟล์หน้ากาก ที่ชื่อว่า 7zjv1v_large.png แล้วนำก็อป path ไปใส่ใน บรรทัดที่ 115
#   โหลด ไฟล์สำหรับ detect face ที่ชื่อว่า haarcascade_frontalface_alt2.xml แล้วก็อป path ไปใส่ใส่ใน บรรทัดที่ 88
#   โหลด ไฟล์สำหรับ detect eyes ที่ชื่อว่า haarcascade_eye_tree_eyeglasses.xml แล้วก็อป path ไปใส่ใน บรรทัดที่ 89

from __future__ import print_function
import math
import cv2 as cv
import argparse
import numpy as np

# ปรับแต่งรูปภาพตามใบหน้า ให้เล็กใหญ่ตามขนาดใบหน้า
def image_tranform(x,y,w,h,frame, img):
    img = cv.resize(img,(w,h))
        
        # rows,cols,channels = img.shape
    roi = frame[y:y+h,x:x+w]
    imggray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret, mask = cv.threshold(imggray, 10, 255, cv.THRESH_BINARY)
    mask_inv = cv.bitwise_not(mask)

    frame_bg = cv.bitwise_and(roi,roi,mask = mask_inv)
    img_fg = cv.bitwise_and(img,img,mask = mask)
    dst = cv.add(frame_bg,img_fg)

    rows,cols,ch = dst.shape
    pts1 = np.float32([[y,x],[y+rows,x],[y,x + cols],[y+rows,x + cols]])
    pts2 = np.float32([[y,x],[y+w,x],[y,x+h],[y+w,x+h]])
    M = cv.getPerspectiveTransform(pts1,pts2)
    dst = cv.warpPerspective(dst,M,(cols,rows))
    frame[y:y+h,x:x+w] = dst

    return frame

def detectAndDisplay(frame,img ,eyes_center_locations):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)
    #-- Detect faces
    faces = face_cascade.detectMultiScale(frame_gray)
    eyes_center_locations = eyes_center_locations 
    for (x,y,w,h) in faces:
        
        
        faceROI = frame_gray[y:y+h,x:x+w]
        #-- In each face, detect eyes
        eyes = eyes_cascade.detectMultiScale(faceROI)
        n = 0
        for (x2,y2,w2,h2) in eyes:
            eye_center = (x + x2 + w2//2, y + y2 + h2//2)

            # เก็บข้อมูลตำแหน่ง ตาดวงแรก
            if n == 0:
                # เก็บค่าไว้ใน list eyes_center_locations
                eyes_center_locations[0] = eye_center
            # เก็บข้อมูลตำแหน่ง ตาดวงที่สอง
            elif n == 1:
                eyes_center_locations[1] = eye_center
            n += 1
        
        eye_left = eyes_center_locations[0]
        eye_right = eyes_center_locations[1]

        # ต้องการให้ตาข้างซ้ายเป็นจุดเริ่มต้นสำหรับใช้หาความเอียงของใบหน้า
        if eye_left[0] > eye_right[0]:
            eye_left = eyes_center_locations[1]
            eye_right = eyes_center_locations[0]
        
        # หา องศาใบหน้าที่เอียงด้วยสูตร พีทาโกรัส
        if eye_right[0] - eye_left[0] != 0: # ป้องกัน ส่วนเป็น 0
            tan_theta = (eye_right[1] - eye_left[1]) / (eye_right[0] - eye_left[0])
        else:
            tan_theta = 0

        theta = math.atan(tan_theta)
        degree = math.degrees(theta)

        rows,cols,_ = img.shape
        
        M = cv.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),-degree,1)
        
        dst = cv.warpAffine(img,M,(cols,rows))

        frame = image_tranform(x,y,w,h,frame, dst)
        

    cv.imshow('Capture - Face detection', frame)

parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
parser.add_argument('--face_cascade', help='Path to face cascade.', default='/Users/phuminsathipchan/Desktop/งานมหาลัย/ฝึกPython/Detect_Face/haarcascade_frontalface_alt2.xml')
parser.add_argument('--eyes_cascade', help='Path to eyes cascade.', default='/Users/phuminsathipchan/Desktop/งานมหาลัย/ฝึกPython/Detect_Face/haarcascade_eye_tree_eyeglasses.xml')
parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)
args = parser.parse_args()
face_cascade_name = args.face_cascade
eyes_cascade_name = args.eyes_cascade
face_cascade = cv.CascadeClassifier()
eyes_cascade = cv.CascadeClassifier()
#-- 1. Load the cascades
if not face_cascade.load(cv.samples.findFile(face_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)
if not eyes_cascade.load(cv.samples.findFile(eyes_cascade_name)):
    print('--(!)Error loading eyes cascade')
    exit(0)
camera_device = args.camera
#-- 2. Read the video stream
cap = cv.VideoCapture(camera_device)
if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)

# list เก็บค่าตา ทั้ง 2 ข้าง
eyes_center_locations = [(0,0),(0,0)]
while True:
    ret, frame = cap.read()
    # รูปหน้ากาก
    img = cv.imread("/Users/phuminsathipchan/Desktop/งานมหาลัย/ฝึกPython/Detect_Face/7zjv1v_large.png")
    if frame is None:
        print('--(!) No captured frame -- Break!')
        break
    detectAndDisplay(frame,img,eyes_center_locations)
    if cv.waitKey(10) == 27:
        break