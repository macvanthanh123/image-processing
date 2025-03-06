import copy
import cv2
import numpy as np
from keras.models import load_model
import time
import tkinter as tk

prediction = ''
score = 0
bgModel = None

gesture_names = {0: 'nam',
                 1: 'L',
                 2: 'ok',
                 3: 'hai',
                 4: 'xoe'}

model = load_model('image-processing/models/mymodel.keras')

def predict_rgb_image_vgg(image):
    image = np.array(image, dtype='float32')
    image /= 255
    pred_array = model.predict(image)
    result = gesture_names[np.argmax(pred_array)]
    score = float("%0.2f" % (max(pred_array[0]) * 100))
    return result, score

def remove_background(frame):
    fgmask = bgModel.apply(frame, learningRate=learningRate)
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res

cap_region_x_begin = 0.5
cap_region_y_end = 0.8

threshold = 60
blurValue = 41
bgSubThreshold = 50
learningRate = 0

predThreshold = 95
isBgCaptured = 0

camera = cv2.VideoCapture(0)
camera.set(10, 200)
camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.01)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while camera.isOpened():
    ret, frame = camera.read()
    frame = cv2.bilateralFilter(frame, 5, 50, 100)
    frame = cv2.flip(frame, 1)
    
    overlay = frame.copy()
    cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                  (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)
    
    cv2.rectangle(frame, (0, 0), (1280, 50), (0, 0, 0), -1)
    
    if isBgCaptured == 1:
        img = remove_background(frame)
        img = img[0:int(cap_region_y_end * frame.shape[0]),
                  int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 10)

        ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((5,5),np.uint8)
        test =  cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        cv2.imshow('Threshold', test)
        
        contours, hierarchy = cv2.findContours(test, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        min_area = 5000  
        hand_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

        if hand_contours:
            max_contour = max(hand_contours, key=cv2.contourArea)  
            x, y, w, h = cv2.boundingRect(max_contour)
            aspect_ratio = w / h

            if 0.5 < aspect_ratio < 2.0:  
                cv2.drawContours(frame, [max_contour + np.array([int(cap_region_x_begin * frame.shape[1]), 0])], -1, (0, 255, 0), 2)
        
        if (np.count_nonzero(test) / (test.shape[0] * test.shape[0]) > 0.2):
            target = np.stack((test,) * 3, axis=-1)
            target = cv2.resize(target, (224, 224))
            target = target.reshape(1, 224, 224, 3)
            prediction, score = predict_rgb_image_vgg(target)

            if score >= predThreshold:
                frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
                cv2.putText(frame, f"Sign: {prediction} ({score}%)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    k = cv2.waitKey(10)
    if k == ord('q'):
        break
    elif k == ord('b'):
        bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
        isBgCaptured = 1
        print('Background captured')
        time.sleep(2)
    elif k == ord('r'):
        bgModel = None
        isBgCaptured = 0
        print('Background reset')
        time.sleep(1)

    cv2.imshow('Hand Gesture Recognition', frame)

cv2.destroyAllWindows()
camera.release()
