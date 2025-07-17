from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
from datetime import datetime
import os

from models import DatabaseManager
from face_detection import FaceDetector
from liveness_detection import LivenessDetector
from config import Config, LIVENESS_METHODS

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 初始化组件
Config.init_directories()
db = DatabaseManager(Config.DATABASE_PATH)
face_detector = FaceDetector(db)
liveness_detector = LivenessDetector()


def base64_to_image(base64_string):
    """将base64字符串转换为OpenCV图像"""
    try:
        if not base64_string:
            print("base64字符串为空")
            return None

        # 移除data:image前缀
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]

        print(f"base64字符串长度: {len(base64_string)}")

        # 解码base64
        try:
            image_data = base64.b64decode(base64_string)
            print(f"解码后数据长度: {len(image_data)}")
        except Exception as e:
            print(f"base64解码失败: {e}")
            return None

        # 转换为numpy数组
        try:
            nparr = np.frombuffer(image_data, np.uint8)
            print(f"numpy数组长度: {len(nparr)}")
        except Exception as e:
            print(f"numpy数组转换失败: {e}")
            return None

        # 解码为图像
        try:
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                print("cv2.imdecode返回None")
                return None

            print(f"图像解码成功，尺寸: {image.shape}")
            return image
        except Exception as e:
            print(f"图像解码失败: {e}")
            return None

    except Exception as e:
        print(f"Base64转换整体错误: {e}")
        return None


def image_to_base64(image):
    """将OpenCV图像转换为base64字符串"""
    try:
        _, buffer = cv2.imencode('.jpg', image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{image_base64}"
    except Exception as e:
        print(f"图像转换错误: {e}")
        return None


@app.route('/')
def index():
    """主页路由"""
    return jsonify({
        "message": "班级考勤系统API",
        "version": "2.0",
        "endpoints": {
            "POST /api/detect_face": "人脸检测和识别",
            "POST /api/liveness_detection": "活体检测",
            "POST /api/attendance": "考勤打卡",
            "POST /api/add_student": "添加学生",
            "DELETE /api/delete_student/<student_id>": "删除学生",
            "GET /api/students": "获取学生列表",
            "GET /api/attendance_records": "获取考勤记录",
            "GET /api/attendance_summary": "获取考勤汇总",
            "GET /api/system_status": "系统状态",
            "GET /api/liveness_methods": "获取活体检测方法"
        }
    })


@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查API"""
    try:
        # 检查数据库连接
        students = db.get_all_students()

        return jsonify({
            "status": "healthy",
            "database": "connected",
            "students_count": len(students),
            "face_detector": "initialized",
            "liveness_detector": "initialized",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500


@app.route('/api/debug/student_add', methods=['POST'])
def debug_student_add():
    """调试学生添加功能"""
    try:
        data = request.get_json()

        debug_info = {
            "received_data": {
                "has_student_id": "student_id" in data,
                "has_name": "name" in data,
                "has_class_name": "class_name" in data,
                "has_image": "image" in data,
                "data_keys": list(data.keys()) if data else []
            },
            "image_info": {},
            "database_info": {},
            "face_detector_info": {}
        }

        if data and "image" in data:
            image_data = data["image"]
            debug_info["image_info"] = {
                "image_type": type(image_data).__name__,
                "image_length": len(image_data) if image_data else 0,
                "has_data_prefix": image_data.startswith("data:") if image_data else False,
                "has_base64_comma": "," in image_data if image_data else False
            }

            # 尝试转换图像
            try:
                image = base64_to_image(image_data)
                debug_info["image_info"]["conversion_success"] = image is not None
                if image is not None:
                    debug_info["image_info"]["image_shape"] = image.shape
            except Exception as e:
                debug_info["image_info"]["conversion_error"] = str(e)

        # 检查数据库
        try:
            students = db.get_all_students()
            debug_info["database_info"]["connection_success"] = True
            debug_info["database_info"]["student_count"] = len(students)
        except Exception as e:
            debug_info["database_info"]["connection_error"] = str(e)

        # 检查人脸检测器
        debug_info["face_detector_info"]["known_faces"] = len(face_detector.known_face_encodings)
        debug_info["face_detector_info"]["baidu_api"] = face_detector.baidu_access_token is not None

        return jsonify({
            "success": True,
            "debug_info": debug_info,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500


@app.route('/api/system_status', methods=['GET'])
def system_status():
    """获取系统状态 - 带详细调试"""
    try:
        print("=== 获取系统状态调试 ===")

        # 逐个获取状态，捕获每个可能的异常
        status_data = {
            "status": "running",
            "timestamp": datetime.now().isoformat()
        }

        # 1. 获取人脸数据库数量
        try:
            face_count = len(face_detector.known_face_encodings)
            status_data["face_database_count"] = face_count
            print(f"✓ 人脸数据库数量: {face_count}")
        except Exception as e:
            print(f"✗ 获取人脸数据库数量失败: {e}")
            status_data["face_database_count"] = 0

        # 2. 获取百度API状态
        try:
            baidu_available = face_detector.baidu_access_token is not None
            status_data["baidu_api_available"] = baidu_available
            print(f"✓ 百度API可用: {baidu_available}")
        except Exception as e:
            print(f"✗ 获取百度API状态失败: {e}")
            status_data["baidu_api_available"] = False

        # 3. 获取dlib状态 - 重点调试
        try:
            print(f"调试: liveness_detector对象类型: {type(liveness_detector)}")
            print(f"调试: liveness_detector是否存在: {liveness_detector is not None}")

            if hasattr(liveness_detector, 'dlib_available'):
                dlib_available = liveness_detector.dlib_available
                print(f"✓ dlib可用状态: {dlib_available}")
            else:
                print("✗ liveness_detector没有dlib_available属性")
                dlib_available = False

            status_data["dlib_available"] = dlib_available
        except Exception as e:
            print(f"✗ 获取dlib状态失败: {e}")
            import traceback
            traceback.print_exc()
            status_data["dlib_available"] = False

        # 4. 获取活体检测方法列表
        try:
            methods = list(LIVENESS_METHODS.keys())
            status_data["liveness_methods"] = methods
            print(f"✓ 活体检测方法: {methods}")
        except Exception as e:
            print(f"✗ 获取活体检测方法失败: {e}")
            status_data["liveness_methods"] = []

        # 5. 获取活体检测阈值
        try:
            threshold = Config.LIVENESS_THRESHOLD
            status_data["liveness_threshold"] = threshold
            print(f"✓ 活体检测阈值: {threshold}")
        except Exception as e:
            print(f"✗ 获取活体检测阈值失败: {e}")
            status_data["liveness_threshold"] = 0.3

        print(f"最终状态数据: {status_data}")
        return jsonify(status_data)

    except Exception as e:
        print(f"system_status整体异常: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500


@app.route('/api/liveness_methods', methods=['GET'])
def get_liveness_methods():
    """获取支持的活体检测方法"""
    methods = []
    for key, info in LIVENESS_METHODS.items():
        methods.append({
            "key": key,
            "name": info['name'],
            "description": info['description']
        })

    return jsonify({
        "success": True,
        "methods": methods,
        "default_method": "combined"
    })


@app.route('/api/detect_face', methods=['POST'])
def detect_face():
    """人脸检测和识别API"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "缺少图像数据"}), 400

        # 转换base64图像
        image = base64_to_image(data['image'])
        if image is None:
            return jsonify({"error": "图像格式错误"}), 400

        # 进行人脸识别
        face_results = face_detector.recognize_face(image)

        # 绘制人脸框（可选）
        annotated_image_base64 = None
        if data.get('draw_boxes', False):
            annotated_image = face_detector.draw_face_boxes(image.copy(), face_results)
            annotated_image_base64 = image_to_base64(annotated_image)

        return jsonify({
            "success": True,
            "face_count": len(face_results),
            "faces": face_results,
            "annotated_image": annotated_image_base64,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        print(f"人脸检测失败: {str(e)}")
        return jsonify({"error": f"人脸检测失败: {str(e)}"}), 500


@app.route('/api/liveness_detection', methods=['POST'])
def liveness_detection():
    """活体检测API"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "缺少图像数据"}), 400

        # 转换base64图像
        image = base64_to_image(data['image'])
        if image is None:
            return jsonify({"error": "图像格式错误"}), 400

        # 获取检测方法，默认使用combined
        method = data.get('method', 'combined')

        # 验证方法是否支持
        if method not in LIVENESS_METHODS:
            return jsonify({"error": f"不支持的活体检测方法: {method}"}), 400

        # 进行活体检测
        liveness_result = liveness_detector.comprehensive_liveness_detection(image, method)

        return jsonify({
            "success": True,
            "liveness_score": liveness_result['final_score'],
            "is_live": liveness_result['is_live'],
            "method": method,
            "method_name": LIVENESS_METHODS[method]['name'],
            "details": liveness_result['details'],
            "message": liveness_result['message'],
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        print(f"活体检测失败: {str(e)}")
        return jsonify({"error": f"活体检测失败: {str(e)}"}), 500


@app.route('/api/attendance', methods=['POST'])
def attendance():
    """考勤打卡API - 完整调试版本"""
    print("\n" + "=" * 50)
    print("考勤请求开始处理")
    print(f"请求方法: {request.method}")
    print(f"请求路径: {request.path}")
    print(f"Content-Type: {request.content_type}")

    try:
        # 1. 获取请求数据
        print("1. 获取请求数据...")
        try:
            data = request.get_json()
            print(f"   - 数据类型: {type(data)}")
            if data:
                print(f"   - 数据键: {list(data.keys())}")
                if 'image' in data:
                    print(f"   - 图像数据长度: {len(data['image'])}")
                    print(f"   - 图像数据前50字符: {data['image'][:50]}...")
            else:
                print("   - 数据为空")
        except Exception as e:
            print(f"   - 获取JSON数据失败: {e}")
            return jsonify({"error": "无法解析请求数据"}), 400

        if not data or 'image' not in data:
            print("2. 错误: 缺少图像数据")
            return jsonify({"error": "缺少图像数据"}), 400

        # 2. 转换base64图像
        print("2. 转换base64图像...")
        try:
            image = base64_to_image(data['image'])
            if image is None:
                print("   - 图像转换失败")
                return jsonify({"error": "图像格式错误"}), 400
            print(f"   - 图像转换成功，尺寸: {image.shape}")
        except Exception as e:
            print(f"   - 图像转换异常: {e}")
            return jsonify({"error": f"图像处理失败: {str(e)}"}), 400

        # 3. 获取检测方法
        liveness_method = data.get('liveness_method', 'combined')
        print(f"3. 活体检测方法: {liveness_method}")

        # 验证方法是否支持
        if liveness_method not in LIVENESS_METHODS:
            print(f"   - 不支持的检测方法: {liveness_method}")
            return jsonify({"error": f"不支持的活体检测方法: {liveness_method}"}), 400

        # 4. 人脸识别
        print("4. 开始人脸识别...")
        try:
            face_results = face_detector.recognize_face(image)
            print(f"   - 检测到 {len(face_results)} 个人脸")
            for i, face in enumerate(face_results):
                print(
                    f"   - 人脸{i + 1}: {face.get('student_name', 'Unknown')} (置信度: {face.get('confidence', 0):.2f})")
        except Exception as e:
            print(f"   - 人脸识别异常: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({"error": f"人脸识别失败: {str(e)}"}), 500

        if not face_results:
            print("   - 未检测到人脸")
            return jsonify({
                "success": False,
                "error": "未检测到人脸",
                "timestamp": datetime.now().isoformat()
            })

        # 5. 活体检测
        print("5. 开始活体检测...")
        try:
            print(f"   - 检测方法: {liveness_method}")
            print(f"   - liveness_detector状态: {type(liveness_detector)}")

            liveness_result = liveness_detector.comprehensive_liveness_detection(image, liveness_method)
            print(f"   - 检测完成，结果: {liveness_result}")

            if not isinstance(liveness_result, dict):
                print(f"   - 活体检测返回值类型错误: {type(liveness_result)}")
                return jsonify({"error": "活体检测返回值格式错误"}), 500

        except Exception as e:
            print(f"   - 活体检测异常: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({"error": f"活体检测失败: {str(e)}"}), 500

        # 检查活体检测结果
        final_score = liveness_result.get('final_score', 0)
        is_live = liveness_result.get('is_live', False)
        print(f"   - 活体分数: {final_score}, 是否通过: {is_live}")

        if not is_live:
            print("   - 活体检测未通过")
            return jsonify({
                "success": False,
                "error": "活体检测失败",
                "liveness_score": final_score,
                "message": liveness_result.get('message', ''),
                "method": liveness_method,
                "timestamp": datetime.now().isoformat()
            })

        # 6. 处理识别结果
        print("6. 处理识别结果...")
        attendance_results = []

        for face_result in face_results:
            student_id = face_result.get('student_id', 'Unknown')
            student_name = face_result.get('student_name', 'Unknown')
            confidence = face_result.get('confidence', 0)

            if student_id != "Unknown":
                print(f"   - 处理学生: {student_name} ({student_id})")

                # 记录考勤
                try:
                    success = db.add_attendance_record(
                        student_id=student_id,
                        student_name=student_name,
                        detection_method=liveness_method,
                        confidence=confidence,
                        liveness_score=final_score
                    )
                    print(f"   - 考勤记录: {'成功' if success else '失败'}")
                except Exception as e:
                    print(f"   - 考勤记录异常: {e}")
                    success = False

                attendance_results.append({
                    "student_id": student_id,
                    "student_name": student_name,
                    "confidence": confidence,
                    "success": success
                })

        if not attendance_results:
            print("   - 未识别到已注册学生")
            return jsonify({
                "success": False,
                "error": "未识别到已注册学生",
                "faces_detected": len(face_results),
                "timestamp": datetime.now().isoformat()
            })

        # 7. 返回成功结果
        result = {
            "success": True,
            "attendance_results": attendance_results,
            "liveness_score": final_score,
            "liveness_method": liveness_method,
            "liveness_method_name": LIVENESS_METHODS.get(liveness_method, {}).get('name', liveness_method),
            "message": "考勤打卡成功",
            "timestamp": datetime.now().isoformat()
        }

        print("7. 考勤处理成功")
        print(f"   - 返回结果: {result}")
        print("=" * 50 + "\n")

        return jsonify(result)

    except Exception as e:
        print(f"考勤处理顶层异常: {e}")
        import traceback
        print("详细异常信息:")
        traceback.print_exc()
        print("=" * 50 + "\n")
        return jsonify({"error": f"服务器内部错误: {str(e)}"}), 500

@app.route('/api/add_student', methods=['POST'])
def add_student():
    """添加学生API"""
    try:
        # 获取JSON数据
        data = request.get_json()
        if not data:
            return jsonify({"error": "请求数据为空"}), 400

        print(f"收到添加学生请求: {data.keys()}")

        # 检查必需字段
        required_fields = ['student_id', 'name', 'class_name', 'image']
        for field in required_fields:
            if field not in data or not data[field]:
                return jsonify({"error": f"缺少必需字段: {field}"}), 400

        # 数据验证
        student_id = str(data['student_id']).strip()
        name = str(data['name']).strip()
        class_name = str(data['class_name']).strip()

        if len(student_id) < 1 or len(student_id) > 20:
            return jsonify({"error": "学号长度应在1-20字符之间"}), 400

        if len(name) < 1 or len(name) > 50:
            return jsonify({"error": "姓名长度应在1-50字符之间"}), 400

        if len(class_name) < 1 or len(class_name) > 30:
            return jsonify({"error": "班级名长度应在1-30字符之间"}), 400

        # 转换base64图像
        print("开始转换图像数据...")
        image = base64_to_image(data['image'])
        if image is None:
            return jsonify({"error": "图像格式错误，请重新拍摄照片"}), 400

        print(f"开始添加学生: {name} ({student_id})")

        # 检查学生是否已存在
        existing_student = db.get_student(student_id)
        if existing_student:
            return jsonify({"error": f"学号 {student_id} 已存在，请使用其他学号"}), 400

        # 先添加学生基本信息到数据库
        print("添加学生基本信息到数据库...")
        db_success, db_message = db.add_student(
            student_id=student_id,
            name=name,
            class_name=class_name,
            photo_path=f"{student_id}_{name}.jpg"
        )

        if not db_success:
            print(f"数据库添加失败: {db_message}")
            return jsonify({"error": db_message}), 400

        print("学生基本信息添加成功，开始处理人脸数据...")

        # 然后添加人脸到检测器（这会同时更新数据库中的人脸编码）
        face_success, face_message = face_detector.add_face_to_database(
            image, student_id, name
        )

        if not face_success:
            print(f"人脸添加失败: {face_message}")
            # 如果人脸添加失败，删除已添加的学生记录
            print("删除已添加的学生记录...")
            db.delete_student(student_id)
            return jsonify({"error": f"人脸处理失败: {face_message}"}), 400

        print(f"学生添加完全成功: {name} ({student_id})")

        return jsonify({
            "success": True,
            "message": "学生添加成功",
            "student_id": student_id,
            "name": name,
            "face_quality_message": face_message,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        print(f"添加学生时发生异常: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"服务器内部错误: {str(e)}"}), 500


@app.route('/api/delete_student/<student_id>', methods=['DELETE'])
def delete_student(student_id):
    """删除学生API"""
    try:
        if not student_id:
            return jsonify({"error": "缺少学生ID"}), 400

        print(f"开始删除学生: {student_id}")

        # 从数据库删除学生
        success, message = db.delete_student(student_id)

        if not success:
            return jsonify({"error": message}), 400

        # 重新加载人脸数据库（从内存中移除该学生的人脸编码）
        face_detector.reload_face_database()

        # 删除人脸图片文件（可选）
        try:
            # 查找并删除对应的人脸图片文件
            import glob
            pattern = os.path.join(Config.FACE_DB_PATH, f"{student_id}_*.jpg")
            files = glob.glob(pattern)
            for file_path in files:
                os.remove(file_path)
                print(f"删除人脸图片文件: {file_path}")
        except Exception as e:
            print(f"删除人脸图片文件时出错: {e}")

        return jsonify({
            "success": True,
            "message": message,
            "student_id": student_id,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        print(f"删除学生失败: {str(e)}")
        return jsonify({"error": f"删除学生失败: {str(e)}"}), 500


@app.route('/api/students', methods=['GET'])
def get_students():
    """获取学生列表API - 修复版本"""
    try:
        print("开始获取学生列表...")
        students = db.get_all_students()
        print(f"从数据库获取到 {len(students)} 个学生")

        student_list = []
        for i, student in enumerate(students):
            try:
                # 确保数据库字段索引正确
                # 数据库字段顺序：id, student_id, name, class_name, face_encoding, photo_path, face_quality_score, is_active, created_at, updated_at
                student_info = {
                    "id": student[0] if len(student) > 0 else None,
                    "student_id": student[1] if len(student) > 1 else "",
                    "name": student[2] if len(student) > 2 else "",
                    "class_name": student[3] if len(student) > 3 else "",
                    "has_face_encoding": bool(student[4]) if len(student) > 4 and student[4] else False,
                    "photo_path": student[5] if len(student) > 5 else None,
                    "face_quality_score": float(student[6]) if len(student) > 6 and student[6] else 0.0,
                    "is_active": bool(student[7]) if len(student) > 7 else True,
                    "created_at": student[8] if len(student) > 8 else None,
                    "updated_at": student[9] if len(student) > 9 else None
                }
                student_list.append(student_info)
                print(f"处理学生 {i + 1}: {student_info['name']} ({student_info['student_id']})")
            except Exception as e:
                print(f"处理学生数据时发生错误: {e}, 学生数据: {student}")
                print(f"学生数据长度: {len(student) if student else 0}")
                # 创建一个基本的学生信息，避免完全跳过
                try:
                    student_info = {
                        "id": student[0] if student and len(student) > 0 else i,
                        "student_id": student[1] if student and len(student) > 1 else f"ERROR_{i}",
                        "name": student[2] if student and len(student) > 2 else f"错误数据_{i}",
                        "class_name": student[3] if student and len(student) > 3 else "未知班级",
                        "has_face_encoding": False,
                        "photo_path": None,
                        "face_quality_score": 0.0,
                        "is_active": True,
                        "created_at": None,
                        "updated_at": None
                    }
                    student_list.append(student_info)
                    print(f"使用错误恢复处理学生 {i + 1}")
                except:
                    print(f"完全跳过学生 {i + 1}")
                    continue

        print(f"成功处理 {len(student_list)} 个学生数据")

        # 确保返回正确的JSON格式，包含success字段
        response_data = {
            "success": True,
            "students": student_list,
            "total_count": len(student_list),
            "timestamp": datetime.now().isoformat(),
            "message": f"成功获取{len(student_list)}个学生信息"
        }

        print(f"返回响应数据: success={response_data['success']}, count={response_data['total_count']}")
        return jsonify(response_data)

    except Exception as e:
        print(f"获取学生列表失败: {str(e)}")
        import traceback
        traceback.print_exc()

        # 返回错误响应，但保持JSON格式一致
        error_response = {
            "success": False,
            "error": f"获取学生列表失败: {str(e)}",
            "students": [],
            "total_count": 0,
            "timestamp": datetime.now().isoformat()
        }
        return jsonify(error_response), 500


@app.route('/api/debug/students', methods=['GET'])
def debug_students():
    """调试学生数据 - 增强版本"""
    try:
        # 获取原始数据库数据
        students_raw = db.get_all_students()

        debug_info = {
            "database_query_result": {
                "count": len(students_raw),
                "first_student_raw": students_raw[0] if students_raw else None,
                "first_student_length": len(students_raw[0]) if students_raw else 0,
                "all_students_sample": students_raw[:3] if len(students_raw) > 0 else []  # 只显示前3个避免数据过多
            }
        }

        # 检查数据库表结构
        import sqlite3
        try:
            conn = sqlite3.connect(db.db_path)
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(students)")
            table_info = cursor.fetchall()
            conn.close()
            debug_info["table_structure"] = table_info
        except Exception as e:
            debug_info["table_structure_error"] = str(e)

        # 尝试手动查询
        try:
            conn = sqlite3.connect(db.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM students WHERE is_active = 1")
            active_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM students")
            total_count = cursor.fetchone()[0]
            conn.close()

            debug_info["manual_query"] = {
                "total_students": total_count,
                "active_students": active_count
            }
        except Exception as e:
            debug_info["manual_query_error"] = str(e)

        return jsonify({
            "success": True,
            "debug_info": debug_info,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        import traceback
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "timestamp": datetime.now().isoformat()
        }), 500


# 如果你的models.py中的get_all_students方法有问题，也可以用这个修复版本替换：
"""
在models.py的DatabaseManager类中，替换get_all_students方法：

def get_all_students(self):
    \"\"\"获取所有激活的学生信息 - 修复版本\"\"\"
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()

    try:
        # 明确指定查询字段，避免索引混乱
        cursor.execute('''
            SELECT id, student_id, name, class_name, face_encoding, 
                   photo_path, face_quality_score, is_active, created_at, updated_at
            FROM students 
            WHERE is_active = 1 
            ORDER BY class_name, name
        ''')
        results = cursor.fetchall()
        print(f"数据库查询返回 {len(results)} 条记录")

        if results and len(results) > 0:
            print(f"第一条记录字段数量: {len(results[0])}")
            print(f"第一条记录内容: {results[0]}")

        return results

    except Exception as e:
        print(f"数据库查询失败: {e}")
        return []
    finally:
        conn.close()
"""


@app.route('/api/attendance_records', methods=['GET'])
def get_attendance_records():
    """获取考勤记录API"""
    try:
        # 获取查询参数
        date = request.args.get('date')

        if date:
            records = db.get_attendance_by_date(date)
        else:
            records = db.get_today_attendance()

        record_list = []
        for record in records:
            record_list.append({
                "id": record[0],
                "student_id": record[1],
                "student_name": record[2],
                "check_time": record[3],
                "detection_method": record[4],
                "confidence": record[5],
                "liveness_score": record[6],
                "status": record[7]
            })

        return jsonify({
            "success": True,
            "records": record_list,
            "total_count": len(record_list),
            "date": date or datetime.now().strftime('%Y-%m-%d'),
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        print(f"获取考勤记录失败: {str(e)}")
        return jsonify({"error": f"获取考勤记录失败: {str(e)}"}), 500


@app.route('/api/attendance_summary', methods=['GET'])
def get_attendance_summary():
    """获取考勤汇总统计API"""
    try:
        date = request.args.get('date')
        summary = db.get_attendance_summary(date)

        return jsonify({
            "success": True,
            "summary": summary,
            "date": date or datetime.now().strftime('%Y-%m-%d'),
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        print(f"获取考勤汇总失败: {str(e)}")
        return jsonify({"error": f"获取考勤汇总失败: {str(e)}"}), 500


@app.route('/api/debug/global_status', methods=['GET'])
def debug_global_status():
    """调试全局对象状态"""
    try:
        debug_info = {
            "timestamp": datetime.now().isoformat()
        }

        # 检查数据库管理器
        try:
            debug_info["db"] = {
                "type": str(type(db)),
                "exists": db is not None,
                "db_path": getattr(db, 'db_path', 'Unknown')
            }
        except Exception as e:
            debug_info["db"] = {"error": str(e)}

        # 检查人脸检测器
        try:
            debug_info["face_detector"] = {
                "type": str(type(face_detector)),
                "exists": face_detector is not None,
                "known_faces": len(getattr(face_detector, 'known_face_encodings', [])),
                "baidu_token": bool(getattr(face_detector, 'baidu_access_token', False))
            }
        except Exception as e:
            debug_info["face_detector"] = {"error": str(e)}

        # 检查活体检测器
        try:
            debug_info["liveness_detector"] = {
                "type": str(type(liveness_detector)),
                "exists": liveness_detector is not None,
                "has_dlib_available": hasattr(liveness_detector, 'dlib_available'),
                "dlib_available": getattr(liveness_detector, 'dlib_available', 'Unknown'),
                "baidu_token": bool(getattr(liveness_detector, 'baidu_access_token', False))
            }
        except Exception as e:
            debug_info["liveness_detector"] = {"error": str(e)}

        # 检查配置
        try:
            debug_info["config"] = {
                "liveness_threshold": getattr(Config, 'LIVENESS_THRESHOLD', 'Unknown'),
                "liveness_methods": list(LIVENESS_METHODS.keys()) if 'LIVENESS_METHODS' in globals() else 'Unknown'
            }
        except Exception as e:
            debug_info["config"] = {"error": str(e)}

        return jsonify(debug_info)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "API端点不存在"}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "服务器内部错误"}), 500


if __name__ == '__main__':
    print("=" * 60)
    print("正在启动班级考勤系统...")
    print("=" * 60)

    # 系统初始化检查
    print("系统组件初始化完成")
    print(f"人脸数据库: {len(face_detector.known_face_encodings)} 个人脸")
    print(f"百度API状态: {'可用' if face_detector.baidu_access_token else '不可用'}")
    print(f"dlib状态: {'可用' if liveness_detector.dlib_available else '不可用'}")
    print(f"活体检测阈值: {Config.LIVENESS_THRESHOLD}")
    print(f"支持的活体检测方法: {list(LIVENESS_METHODS.keys())}")

    print("=" * 60)
    print("服务器运行在: http://localhost:5000")
    print("API文档: http://localhost:5000")
    print("=" * 60)

    app.run(host='0.0.0.0', port=5000, debug=True)