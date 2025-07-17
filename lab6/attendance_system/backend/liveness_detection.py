import cv2
import numpy as np
import requests
import base64
import dlib
from scipy.spatial import distance as dist
from config import Config
import time


class LivenessDetector:
    def __init__(self):
        self.baidu_access_token = None

        # 眨眼检测相关参数
        self.EYE_AR_THRESH = 0.25  # 眼睛长宽比阈值
        self.EYE_AR_CONSEC_FRAMES = 3  # 连续帧数
        self.COUNTER = 0
        self.TOTAL_BLINKS = 0

        # 嘴部检测相关参数
        self.MOUTH_AR_THRESH = 0.7  # 嘴部长宽比阈值
        self.mouth_opened = False
        self.mouth_open_frames = 0

        # 头部姿态相关参数
        self.head_poses = []
        self.pose_change_threshold = 15  # 角度变化阈值

        # 初始化dlib人脸关键点检测器
        try:
            self.detector = dlib.get_frontal_face_detector()
            # 需要下载shape_predictor_68_face_landmarks.dat文件
            self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
            self.dlib_available = True
            print("dlib初始化成功")
        except Exception as e:
            print(f"警告: dlib人脸关键点检测器初始化失败: {e}")
            self.dlib_available = False

        # 时间戳记录
        self.start_time = time.time()

        self.get_baidu_access_token()

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
                print(f"获取百度访问令牌失败: {result}")
                return None
        except Exception as e:
            print(f"请求百度访问令牌时发生错误: {e}")
            return None

    def image_to_base64(self, image):
        """将图像转换为base64编码"""
        _, buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(buffer).decode('utf-8')

    # 替换你的 baidu_liveness_detection 方法为以下版本：

    def baidu_liveness_detection(self, image):
        """使用百度API进行活体检测 - 官方文档格式"""
        print("=== 百度API活体检测（官方格式）===")

        if not self.baidu_access_token:
            print("百度API未配置，尝试重新获取token")
            self.get_baidu_access_token()
            if not self.baidu_access_token:
                return 0.0, "百度API未配置"

        # 🔧 根据官方文档，URL应该是这个
        request_url = "https://aip.baidubce.com/rest/2.0/face/v3/faceverify"
        request_url = request_url + "?access_token=" + self.baidu_access_token

        print(f"请求URL: {request_url}")

        # 转换图像为base64
        image_base64 = self.image_to_base64(image)
        print(f"图像转换完成，长度: {len(image_base64)}")

        # 🔧 关键：按照官方文档的参数格式
        # 需要构建一个JSON数组的字符串，包含图片信息
        params_array = [
            {
                "image": image_base64,
                "image_type": "BASE64",
                "face_field": "age,beauty,expression"  # 可选字段
            }
        ]

        # 🔧 重要：将数组转换为JSON字符串
        import json
        params = json.dumps(params_array, ensure_ascii=False)

        print(f"参数格式: JSON数组字符串")
        print(f"参数长度: {len(params)}")
        print(f"参数结构: {[{k: '...' if k == 'image' else v for k, v in params_array[0].items()}]}")

        # 🔧 按照官方文档的请求头
        headers = {
            'content-type': 'application/json'  # 注意这里是小写的content-type
        }

        try:
            print("发送活体检测请求...")

            # 🔧 重要：使用data参数而不是json参数
            response = requests.post(request_url, data=params, headers=headers, timeout=20)

            print(f"响应状态码: {response.status_code}")
            print(f"响应内容: {response.text}")

            # 解析响应
            result = response.json()
            print(f"解析的JSON: {result}")

            # 检查API调用是否成功
            if result.get('error_code') == 0:
                face_liveness = result.get('result', {}).get('face_liveness', 0.0)

                print(f"✅ 活体检测成功")
                print(f"   活体分数: {face_liveness}")

                return face_liveness, f"百度活体检测成功: {face_liveness:.3f}"

            else:
                # 处理错误
                error_code = result.get('error_code', 'Unknown')
                error_msg = result.get('error_msg', '未知错误')

                print(f"❌ 百度API调用失败:")
                print(f"   错误代码: {error_code}")
                print(f"   错误信息: {error_msg}")

                return 0.0, f"百度API错误[{error_code}]: {error_msg}"

        except requests.exceptions.Timeout:
            print("❌ 请求超时")
            return 0.0, "百度API请求超时"

        except requests.exceptions.RequestException as e:
            print(f"❌ 请求异常: {e}")
            return 0.0, f"百度API网络错误: {str(e)}"

        except ValueError as e:
            print(f"❌ JSON解析失败: {e}")
            print(f"原始响应: {response.text}")
            return 0.0, "百度API响应格式错误"

        except Exception as e:
            print(f"❌ 其他异常: {e}")
            import traceback
            traceback.print_exc()
            return 0.0, f"百度API调用异常: {str(e)}"
    def eye_aspect_ratio(self, eye):
        """计算眼睛长宽比"""
        # 计算垂直方向的眼睛距离
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])

        # 计算水平方向的眼睛距离
        C = dist.euclidean(eye[0], eye[3])

        # 计算眼睛长宽比
        ear = (A + B) / (2.0 * C)
        return ear

    def mouth_aspect_ratio(self, mouth):
        """计算嘴部长宽比"""
        # 计算垂直方向的嘴部距离
        A = dist.euclidean(mouth[2], mouth[10])  # 51, 59
        B = dist.euclidean(mouth[4], mouth[8])  # 53, 57

        # 计算水平方向的嘴部距离
        C = dist.euclidean(mouth[0], mouth[6])  # 49, 55

        # 计算嘴部长宽比
        mar = (A + B) / (2.0 * C)
        return mar

    def blink_detection(self, image):
        """眨眼检测"""
        if not self.dlib_available:
            return 0.5, "dlib不可用，返回默认分数"

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)

        if len(faces) == 0:
            return 0.0, "未检测到人脸"

        for face in faces:
            landmarks = self.predictor(gray, face)
            landmarks_np = np.array([[p.x, p.y] for p in landmarks.parts()])

            # 提取左右眼的关键点
            left_eye = landmarks_np[36:42]
            right_eye = landmarks_np[42:48]

            # 计算眼睛长宽比
            left_ear = self.eye_aspect_ratio(left_eye)
            right_ear = self.eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0

            # 检查是否眨眼
            if ear < self.EYE_AR_THRESH:
                self.COUNTER += 1
            else:
                if self.COUNTER >= self.EYE_AR_CONSEC_FRAMES:
                    self.TOTAL_BLINKS += 1
                self.COUNTER = 0

            # 计算活体分数（基于眨眼次数和时间）
            elapsed_time = time.time() - self.start_time
            if elapsed_time > 2:  # 至少观察2秒
                blink_rate = self.TOTAL_BLINKS / elapsed_time
                # 正常眨眼频率大约每分钟15-20次，即每秒0.25-0.33次
                if 0.1 <= blink_rate <= 0.5:
                    liveness_score = min(0.9, 0.5 + blink_rate)
                else:
                    liveness_score = 0.3
            else:
                liveness_score = 0.5

            return liveness_score, f"眨眼检测: {self.TOTAL_BLINKS}次眨眼，评分{liveness_score:.2f}"

        return 0.0, "人脸处理失败"

    def mouth_detection(self, image):
        """张嘴检测"""
        if not self.dlib_available:
            return 0.5, "dlib不可用，返回默认分数"

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)

        if len(faces) == 0:
            return 0.0, "未检测到人脸"

        for face in faces:
            landmarks = self.predictor(gray, face)
            landmarks_np = np.array([[p.x, p.y] for p in landmarks.parts()])

            # 提取嘴部关键点 (49-68)
            mouth = landmarks_np[48:68]

            # 计算嘴部长宽比
            mar = self.mouth_aspect_ratio(mouth)

            # 检查是否张嘴
            if mar > self.MOUTH_AR_THRESH:
                if not self.mouth_opened:
                    self.mouth_opened = True
                    self.mouth_open_frames = 1
                else:
                    self.mouth_open_frames += 1
            else:
                if self.mouth_opened and self.mouth_open_frames >= 3:
                    # 检测到有效的张嘴动作
                    liveness_score = 0.8
                    return liveness_score, f"检测到张嘴动作，评分{liveness_score:.2f}"
                self.mouth_opened = False
                self.mouth_open_frames = 0

            # 基于当前状态返回分数
            if self.mouth_opened:
                return 0.6, "正在张嘴"
            else:
                return 0.4, "嘴部闭合"

        return 0.0, "人脸处理失败"

    def head_pose_detection(self, image):
        """头部姿态检测"""
        if not self.dlib_available:
            return 0.5, "dlib不可用，返回默认分数"

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)

        if len(faces) == 0:
            return 0.0, "未检测到人脸"

        for face in faces:
            landmarks = self.predictor(gray, face)
            landmarks_np = np.array([[p.x, p.y] for p in landmarks.parts()])

            # 3D模型点
            model_points = np.array([
                (0.0, 0.0, 0.0),  # 鼻尖
                (0.0, -330.0, -65.0),  # 下巴
                (-225.0, 170.0, -135.0),  # 左眼左角
                (225.0, 170.0, -135.0),  # 右眼右角
                (-150.0, -150.0, -125.0),  # 左嘴角
                (150.0, -150.0, -125.0)  # 右嘴角
            ])

            # 2D图像点
            image_points = np.array([
                landmarks_np[30],  # 鼻尖
                landmarks_np[8],  # 下巴
                landmarks_np[36],  # 左眼左角
                landmarks_np[45],  # 右眼右角
                landmarks_np[48],  # 左嘴角
                landmarks_np[54]  # 右嘴角
            ], dtype="double")

            # 相机参数
            size = image.shape
            focal_length = size[1]
            center = (size[1] / 2, size[0] / 2)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype="double")

            dist_coeffs = np.zeros((4, 1))

            # 求解PnP问题
            success, rotation_vector, translation_vector = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if success:
                # 将旋转向量转换为欧拉角
                rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
                angles = cv2.RQDecomp3x3(rotation_matrix)[0]

                # 记录头部姿态
                self.head_poses.append(angles)

                # 保持最近10个姿态记录
                if len(self.head_poses) > 10:
                    self.head_poses.pop(0)

                # 计算姿态变化
                if len(self.head_poses) >= 2:
                    pose_diff = np.abs(np.array(self.head_poses[-1]) - np.array(self.head_poses[0]))
                    max_diff = np.max(pose_diff)

                    if max_diff > self.pose_change_threshold:
                        liveness_score = min(0.9, 0.5 + max_diff / 100)
                        return liveness_score, f"检测到头部移动: {max_diff:.1f}度，评分{liveness_score:.2f}"
                    else:
                        return 0.4, "头部相对静止"
                else:
                    return 0.5, "正在分析头部姿态"

        return 0.0, "姿态检测失败"

    def dlib_comprehensive_detection(self, image):
        """dlib综合活体检测（眨眼40% + 张嘴40% + 姿态20%）"""
        if not self.dlib_available:
            return 0.0, "dlib不可用"

        # 获取各项检测结果
        blink_score, blink_msg = self.blink_detection(image)
        mouth_score, mouth_msg = self.mouth_detection(image)
        pose_score, pose_msg = self.head_pose_detection(image)

        # 按权重计算综合分数
        weights = Config.DLIB_WEIGHTS
        final_score = (
                blink_score * weights['blink'] +
                mouth_score * weights['mouth'] +
                pose_score * weights['pose']
        )

        message = f"dlib综合检测: 眨眼{blink_score:.2f}({weights['blink'] * 100}%), 张嘴{mouth_score:.2f}({weights['mouth'] * 100}%), 姿态{pose_score:.2f}({weights['pose'] * 100}%)"

        return final_score, message

    def comprehensive_liveness_detection(self, image, method='combined'):
        """
        综合活体检测
        method可选值：
        - 'baidu_only': 仅使用百度API
        - 'dlib_only': 仅使用dlib检测（眨眼40%+张嘴40%+姿态20%）
        - 'combined': 百度API 50% + dlib检测 50%
        """
        results = {}
        final_score = 0.0
        messages = []

        print(f"开始活体检测，方法: {method}")

        if method == 'baidu_only':
            # 仅使用百度API
            score, msg = self.baidu_liveness_detection(image)
            results['baidu_api'] = {'score': score, 'message': msg}
            final_score = score
            messages.append(f"百度API: {msg}")

        elif method == 'dlib_only':
            # 仅使用dlib检测
            score, msg = self.dlib_comprehensive_detection(image)
            results['dlib_detection'] = {'score': score, 'message': msg}
            final_score = score
            messages.append(f"dlib检测: {msg}")

        elif method == 'combined':
            # 综合检测：百度API 50% + dlib检测 50%
            baidu_score, baidu_msg = self.baidu_liveness_detection(image)
            dlib_score, dlib_msg = self.dlib_comprehensive_detection(image)

            results['baidu_api'] = {'score': baidu_score, 'message': baidu_msg}
            results['dlib_detection'] = {'score': dlib_score, 'message': dlib_msg}

            # 50% - 50% 权重
            final_score = baidu_score * 0.5 + dlib_score * 0.5

            messages.append(f"百度API(50%): {baidu_msg}")
            messages.append(f"dlib检测(50%): {dlib_msg}")
            messages.append(f"综合评分: {final_score:.2f}")

        else:
            return {
                'final_score': 0.0,
                'is_live': False,
                'details': {},
                'message': f'不支持的检测方法: {method}'
            }

        is_live = final_score >= Config.LIVENESS_THRESHOLD

        print(f"活体检测完成，最终分数: {final_score:.2f}, 是否为活体: {is_live}")

        return {
            'final_score': final_score,
            'is_live': is_live,
            'details': results,
            'message': '; '.join(messages),
            'method': method
        }

    def reset_counters(self):
        """重置计数器"""
        self.COUNTER = 0
        self.TOTAL_BLINKS = 0
        self.mouth_opened = False
        self.mouth_open_frames = 0
        self.head_poses = []
        self.start_time = time.time()
        print("活体检测计数器已重置")