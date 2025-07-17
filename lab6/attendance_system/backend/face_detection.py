import face_recognition
import cv2
import numpy as np
import requests
import base64
import os
from PIL import Image
from config import Config


class FaceDetector:
    def __init__(self, db_manager=None):
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_ids = []
        self.baidu_access_token = None
        self.db_manager = db_manager  # 数据库管理器

        print("初始化人脸检测器...")
        self.get_baidu_access_token()
        self.load_face_database()

    def get_baidu_access_token(self):
        """获取百度API访问令牌"""
        if not Config.BAIDU_API_KEY or not Config.BAIDU_SECRET_KEY:
            print("警告: 百度API密钥未配置")
            return None

        url = Config.BAIDU_TOKEN_URL
        params = {
            'grant_type': 'client_credentials',
            'client_id': Config.BAIDU_API_KEY,
            'client_secret': Config.BAIDU_SECRET_KEY
        }

        try:
            response = requests.post(url, params=params)
            result = response.json()

            if 'access_token' in result:
                self.baidu_access_token = result['access_token']
                print("百度API访问令牌获取成功")
                return self.baidu_access_token
            else:
                print(f"获取访问令牌失败: {result}")
                return None
        except Exception as e:
            print(f"请求访问令牌时发生错误: {e}")
            return None

    def image_to_base64(self, image):
        """将图像转换为base64编码"""
        if isinstance(image, np.ndarray):
            # OpenCV图像 (BGR)
            _, buffer = cv2.imencode('.jpg', image)
            return base64.b64encode(buffer).decode('utf-8')
        else:
            # PIL图像
            import io
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG')
            return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def detect_face_baidu(self, image):
        """使用百度API检测人脸"""
        if not self.baidu_access_token:
            return None

        url = Config.BAIDU_FACE_DETECT_URL + f"?access_token={self.baidu_access_token}"

        # 转换图像为base64
        image_base64 = self.image_to_base64(image)

        data = {
            'image': image_base64,
            'image_type': 'BASE64',
            'face_field': 'age,beauty,expression,face_shape,gender,glasses,landmark,race,quality,face_type'
        }

        try:
            response = requests.post(url, data=data)
            result = response.json()

            if result.get('error_code') == 0:
                return result['result']
            else:
                print(f"百度人脸检测失败: {result}")
                return None
        except Exception as e:
            print(f"百度人脸检测请求失败: {e}")
            return None

    def load_face_database(self):
        """加载人脸数据库 - 优先从数据库加载，其次从文件"""
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_ids = []

        # 方法1: 从数据库加载人脸编码
        if self.db_manager:
            try:
                face_data = self.db_manager.get_all_face_encodings()
                for data in face_data:
                    self.known_face_encodings.append(data['encoding'])
                    self.known_face_names.append(data['name'])
                    self.known_face_ids.append(data['student_id'])

                if face_data:
                    print(f"从数据库加载了 {len(face_data)} 个人脸编码")
                    return
                else:
                    print("数据库中没有人脸编码数据")
            except Exception as e:
                print(f"从数据库加载人脸编码失败: {e}")

        # 方法2: 从文件系统加载（备用方案）
        print("正在从文件系统加载人脸数据...")
        face_db_path = Config.FACE_DB_PATH
        if not os.path.exists(face_db_path):
            os.makedirs(face_db_path)
            print("人脸数据库目录不存在，已创建")
            return

        loaded_count = 0
        for filename in os.listdir(face_db_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                # 文件名格式: studentid_name.jpg
                name_parts = filename.split('_')
                if len(name_parts) >= 2:
                    student_id = name_parts[0]
                    student_name = '_'.join(name_parts[1:]).replace('.jpg', '').replace('.png', '').replace('.jpeg', '')

                    # 加载图像并编码
                    image_path = os.path.join(face_db_path, filename)
                    try:
                        image = face_recognition.load_image_file(image_path)
                        face_encodings = face_recognition.face_encodings(image)

                        if len(face_encodings) > 0:
                            face_encoding = face_encodings[0]
                            self.known_face_encodings.append(face_encoding)
                            self.known_face_names.append(student_name)
                            self.known_face_ids.append(student_id)

                            # 同时更新数据库中的人脸编码
                            if self.db_manager:
                                quality_score = self.get_face_quality_score_from_encoding(image)
                                self.db_manager.update_student_face_encoding(
                                    student_id, face_encoding, quality_score
                                )

                            loaded_count += 1
                            print(f"加载人脸: {student_name} ({student_id})")
                        else:
                            print(f"未在图片中检测到人脸: {filename}")
                    except Exception as e:
                        print(f"加载人脸图片失败 {filename}: {e}")

        print(f"人脸数据库加载完成，共{loaded_count}个人脸")

    def add_face_to_database(self, image, student_id, student_name):
        """添加人脸到数据库（包括文件和数据库存储）"""
        try:
            print(f"开始添加人脸: {student_name} ({student_id})")

            # 检测人脸
            face_encodings = face_recognition.face_encodings(image)

            if len(face_encodings) == 0:
                return False, "未检测到人脸"

            if len(face_encodings) > 1:
                return False, "检测到多个人脸，请确保图像中只有一个人脸"

            face_encoding = face_encodings[0]

            # 计算人脸质量分数
            quality_score = self.get_face_quality_score(image)

            # 保存图像到文件系统（备份）
            filename = f"{student_id}_{student_name}.jpg"
            image_path = os.path.join(Config.FACE_DB_PATH, filename)

            # 转换为PIL图像并保存
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            pil_image.save(image_path)
            print(f"人脸图片保存到: {image_path}")

            # 更新数据库中的人脸编码
            if self.db_manager:
                success = self.db_manager.update_student_face_encoding(
                    student_id, face_encoding, quality_score
                )
                if not success:
                    return False, "更新数据库中的人脸编码失败"

            # 添加到内存数据库
            # 检查是否已存在该学生的人脸编码
            if student_id in self.known_face_ids:
                # 更新现有编码
                index = self.known_face_ids.index(student_id)
                self.known_face_encodings[index] = face_encoding
                self.known_face_names[index] = student_name
                print(f"更新现有人脸编码: {student_name} ({student_id})")
            else:
                # 添加新编码
                self.known_face_encodings.append(face_encoding)
                self.known_face_names.append(student_name)
                self.known_face_ids.append(student_id)
                print(f"添加新人脸编码: {student_name} ({student_id})")

            return True, f"人脸添加成功，质量分数: {quality_score:.2f}"

        except Exception as e:
            print(f"添加人脸时发生错误: {str(e)}")
            return False, f"添加人脸时发生错误: {str(e)}"

    def get_face_quality_score_from_encoding(self, image):
        """从图像计算人脸质量分数"""
        try:
            # 简单的质量评估：基于图像清晰度和人脸大小
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            # 计算拉普拉斯方差（衡量清晰度）
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

            # 检测人脸大小
            face_locations = face_recognition.face_locations(image)
            if face_locations:
                top, right, bottom, left = face_locations[0]
                face_size = (bottom - top) * (right - left)
                size_score = min(1.0, face_size / (100 * 100))  # 标准化到0-1
            else:
                size_score = 0.5

            # 综合质量分数
            clarity_score = min(1.0, laplacian_var / 1000)  # 标准化清晰度分数
            quality_score = (clarity_score * 0.7 + size_score * 0.3)

            return quality_score
        except Exception as e:
            print(f"计算人脸质量分数时出错: {e}")
            return 0.8  # 默认分数

    def recognize_face(self, image):
        """识别人脸"""
        try:
            # 使用face_recognition库识别
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 查找所有人脸位置和编码
            face_locations = face_recognition.face_locations(rgb_image)
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

            results = []

            for face_encoding, face_location in zip(face_encodings, face_locations):
                # 与已知人脸进行比较
                matches = face_recognition.compare_faces(
                    self.known_face_encodings,
                    face_encoding,
                    tolerance=Config.FACE_RECOGNITION_TOLERANCE
                )

                face_distances = face_recognition.face_distance(
                    self.known_face_encodings,
                    face_encoding
                )

                student_id = "Unknown"
                student_name = "未知"
                confidence = 0.0

                if len(matches) > 0 and True in matches:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        student_id = self.known_face_ids[best_match_index]
                        student_name = self.known_face_names[best_match_index]
                        confidence = 1 - face_distances[best_match_index]

                # 计算人脸框坐标
                top, right, bottom, left = face_location

                results.append({
                    'student_id': student_id,
                    'student_name': student_name,
                    'confidence': confidence,
                    'face_location': {
                        'top': top,
                        'right': right,
                        'bottom': bottom,
                        'left': left
                    }
                })

            return results

        except Exception as e:
            print(f"人脸识别时发生错误: {str(e)}")
            return []

    def draw_face_boxes(self, image, face_results):
        """在图像上绘制人脸框和标签"""
        try:
            for result in face_results:
                location = result['face_location']
                top, right, bottom, left = location['top'], location['right'], location['bottom'], location['left']

                # 绘制人脸框
                color = (0, 255, 0) if result['student_id'] != "Unknown" else (0, 0, 255)
                cv2.rectangle(image, (left, top), (right, bottom), color, 2)

                # 绘制标签背景
                cv2.rectangle(image, (left, bottom - 35), (right, bottom), color, cv2.FILLED)

                # 绘制文字
                font = cv2.FONT_HERSHEY_DUPLEX
                text = f"{result['student_name']} ({result['confidence']:.2f})"
                cv2.putText(image, text, (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1)

            return image

        except Exception as e:
            print(f"绘制人脸框时发生错误: {str(e)}")
            return image

    def get_face_quality_score(self, image):
        """获取人脸质量评分（优先使用百度API）"""
        # 尝试使用百度API获取质量分数
        result = self.detect_face_baidu(image)
        if result and 'face_list' in result and len(result['face_list']) > 0:
            face_info = result['face_list'][0]
            if 'quality' in face_info:
                quality = face_info['quality']
                # 计算综合质量分数
                total_score = (
                        quality.get('blur', 1) * 0.3 +
                        quality.get('illumination', 1) * 0.3 +
                        quality.get('completeness', 1) * 0.4
                )
                print(f"百度API质量分数: {total_score:.2f}")
                return total_score

        # 备用质量评估方法
        score = self.get_face_quality_score_from_encoding(image)
        print(f"本地质量分数: {score:.2f}")
        return score

    def reload_face_database(self):
        """重新加载人脸数据库"""
        print("重新加载人脸数据库...")
        self.load_face_database()
        print(f"人脸数据库重新加载完成，当前共有 {len(self.known_face_encodings)} 个人脸")
