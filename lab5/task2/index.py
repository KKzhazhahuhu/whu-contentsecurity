import threading
import cv2
import os
import json
import dlib
from deepface import DeepFace

# Load the detector
detector = dlib.get_frontal_face_detector()

# Load the predictor
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


class OpcvCapture(threading.Thread):
    def __init__(self, win_name, cam_name):
        super().__init__()
        self.cam_name = cam_name
        self.win_name = win_name
        self.count = 0
        # 存储最近的分析结果
        self.last_results = []

    def run(self):
        capture = cv2.VideoCapture(self.cam_name)
        while (True):
            # 获取一帧
            ret, frame = capture.read()

            if not ret:
                print("Failed to grab frame")
                continue

            display_frame = frame.copy()
            for result in self.last_results:
                face_rect = result["face_rect"]
                age = result["age"]
                emotion = result["emotion"]

                # 绘制人脸框
                cv2.rectangle(display_frame,
                              (face_rect[0], face_rect[1]),
                              (face_rect[2], face_rect[3]),
                              (255, 0, 0), 2)

                # 显示年龄和情绪
                cv2.putText(display_frame, f"Age: {age}",
                            (face_rect[0], face_rect[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Emotion: {emotion}",
                            (face_rect[0], face_rect[1] - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # 绘制面部特征点
                if "landmarks" in result:
                    for point in result["landmarks"]:
                        cv2.circle(display_frame, point, 1, (0, 0, 255), -1)

            # 每10帧处理一次面部检测和分析
            if self.count % 10 == 0:
                gray = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)
                faces = detector(frame)

                # 清除之前的结果
                self.last_results = []

                for face in faces:
                    landmarks = predictor(gray, face)
                    landmark_points = []
                    for n in range(0, 68):
                        x = landmarks.part(n).x
                        y = landmarks.part(n).y
                        landmark_points.append((x, y))

                    # 确保人脸坐标有效
                    if (face.top() < face.bottom() and face.left() < face.right() and
                            face.top() >= 0 and face.left() >= 0 and
                            face.bottom() < frame.shape[0] and face.right() < frame.shape[1]):

                        face_image = frame[face.top():face.bottom(), face.left():face.right()]

                        # 检查face_image是否为空
                        if face_image.size > 0:
                            try:
                                # 添加enforce_detection=False以避免错误
                                result = DeepFace.analyze(face_image, actions=['age', 'emotion'],
                                                          enforce_detection=False)

                                age = result[0]['age']
                                emotion = result[0]['dominant_emotion']

                                # 存储结果以便在后续帧中使用
                                self.last_results.append({
                                    "face_rect": (face.left(), face.top(), face.right(), face.bottom()),
                                    "age": age,
                                    "emotion": emotion,
                                    "landmarks": landmark_points
                                })

                                # 在当前帧上绘制结果
                                cv2.rectangle(display_frame,
                                              (face.left(), face.top()),
                                              (face.right(), face.bottom()),
                                              (255, 0, 0), 2)
                                cv2.putText(display_frame, f"Age: {age}",
                                            (face.left(), face.top() - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                                cv2.putText(display_frame, f"Emotion: {emotion}",
                                            (face.left(), face.top() - 40),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                                # 绘制面部特征点
                                for point in landmark_points:
                                    cv2.circle(display_frame, point, 1, (0, 0, 255), -1)

                            except Exception as e:
                                print(f"DeepFace error: {e}")
                        else:
                            print("Extracted face region is empty")

            cv2.imshow(self.win_name, display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            self.count += 1

        # 释放资源
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    camera1 = OpcvCapture("camera1", 0)
    camera1.start()