#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :index.py
# @Time      :2022/4/2 16:23
# @Author    :bd17kaka

import threading
import cv2
import dlib

detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


class OpcvCapture(threading.Thread):
    def __init__(self, win_name, cam_name):
        super().__init__()
        self.cam_name = cam_name
        self.win_name = win_name

    def run(self):
        # capture = cv2.VideoCapture(self.cam_name)
        capture = cv2.VideoCapture(0)

        while (True):
            # 获取一帧
            ret, frame = capture.read()

            gray = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)
            faces = detector(frame)

            for face in faces:
                landmarks = predictor(gray, face)
                for n in range(0, 68):
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
                    cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

            cv2.imshow(self.win_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


if __name__ == "__main__":
    camera1 = OpcvCapture("Face", 0)
    camera1.start()