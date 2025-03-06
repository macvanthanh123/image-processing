import tkinter as tk
import cv2
from PIL import Image, ImageTk
import mediapipe as mp
import math

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=1)

tipIds = [4, 8, 12, 16, 20]

def distance(p1, p2):
    return math.sqrt((p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)

width, height = 400, 300
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

app = tk.Tk()
app.title("Nhận diện cử chỉ tay")
app.resizable(False, False)

label_widget = tk.Label(app)
label_widget.pack()

gesture_label = tk.Label(app, text="", font=("Arial", 20), fg="blue")
gesture_label.pack()

def open_camera():
    if cap.isOpened():
        success, img = cap.read()
        if success:
            img = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)
            img.flags.writeable = False
            results = hands.process(img)
            img.flags.writeable = True

            lmList = []
            handType = []

            if results.multi_hand_landmarks:
                for hand in results.multi_handedness:
                    hand_label = hand.classification[0].label
                    handType.append(hand_label)

                for hand_landmarks in results.multi_hand_landmarks:
                    myHands = results.multi_hand_landmarks[0]
                    for id, lm in enumerate(myHands.landmark):
                        h, w, c = img.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        lmList.append([id, cx, cy])
                    mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            fingers = []
            if len(lmList) != 0:
                if handType[0] == 'Left':
                    fingers.append(1 if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1] else 0)
                else:  
                    fingers.append(1 if lmList[tipIds[0]][1] < lmList[tipIds[0] - 1][1] else 0)

                for id in range(1, 5):
                    fingers.append(1 if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2] else 0)

                thumb_index_distance = distance(lmList[tipIds[0]], lmList[tipIds[1]])

                if fingers == [0, 0, 0, 0, 0]:  
                    gesture_label.config(text="✊ Nắm tay")

                elif fingers == [1, 1, 0, 0, 0]:  
                    gesture_label.config(text="L")

                elif fingers == [0, 1, 1, 0, 0]:  
                    gesture_label.config(text="✌️")

                elif fingers == [1, 1, 1, 1, 1]:  
                    gesture_label.config(text="🖐️")

                elif thumb_index_distance < 30 and fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 1:  
                    gesture_label.config(text="👌")

                else:
                    gesture_label.config(text="")

            img = ImageTk.PhotoImage(image=Image.fromarray(img))
            label_widget['image'] = img
            label_widget.image = img

    label_widget.after(10, open_camera)

open_camera()

app.bind('<Escape>', lambda e: app.quit())
app.mainloop()
