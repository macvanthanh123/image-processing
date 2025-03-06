import cv2
import numpy as np
from keras.models import load_model
import time

gesture_names = {0: 'Nam', 1: 'L', 2: 'OK', 3: 'Hai', 4: 'Xoe'}
model = load_model('image-processing/models/mymodel.keras')

bgModel = cv2.createBackgroundSubtractorMOG2(0, 50)  
cap_region_x_begin, cap_region_y_end = 0.5, 0.8
threshold, blurValue = 60, 41
predThreshold = 98
detect_timeout = 0.01  

last_detect_time = time.time()

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow('Camera', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Camera', 1280, 720)

def predict_gesture(image):
    image = np.array(image, dtype='float32') / 255.0
    pred_array = model.predict(image)
    result = gesture_names[np.argmax(pred_array)]
    score = round(float(max(pred_array[0]) * 100), 2)
    return result, score

while camera.isOpened():
    ret, frame = camera.read()
    if not ret:
        break

    frame = cv2.flip(cv2.bilateralFilter(frame, 5, 50, 100), 1)
    
    x1, y1, x2, y2 = int(cap_region_x_begin * frame.shape[1]), 0, frame.shape[1], int(cap_region_y_end * frame.shape[0])
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    fgmask = bgModel.apply(frame)
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    img = cv2.bitwise_and(frame, frame, mask=fgmask)[y1:y2, x1:x2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
    _, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Áp dụng biến đổi hình thái
    morph_kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, morph_kernel)  # Loại bỏ nhiễu
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, morph_kernel)  # Lấp đầy khoảng trống
    
    cv2.imshow('Processed Threshold', cv2.resize(thresh, dsize=None, fx=0.5, fy=0.5))

    if np.count_nonzero(thresh) / (thresh.shape[0] * thresh.shape[1]) > 0.2:
        last_detect_time = time.time()
        target = np.stack((thresh,) * 3, axis=-1)
        target = cv2.resize(target, (224, 224)).reshape(1, 224, 224, 3)
        prediction, score = predict_gesture(target)
        print(score)
        if score >= predThreshold:
            cv2.putText(frame, f"Sign: {prediction}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)
        else:
            cv2.putText(frame, "Unknown", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
    else:
        if time.time() - last_detect_time > detect_timeout:
            bgModel = cv2.createBackgroundSubtractorMOG2(0, 50)
            last_detect_time = time.time()
            print("Cập nhật lại nền")

    cv2.imshow('Camera', frame)

    if cv2.waitKey(10) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()