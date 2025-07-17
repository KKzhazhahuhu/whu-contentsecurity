import os


class Config:
    # 数据库配置
    DATABASE_PATH = '../database/attendance.db'

    # 人脸识别配置
    FACE_DB_PATH = 'static/face_db/'
    FACE_RECOGNITION_TOLERANCE = 0.5

    # 百度云人脸识别API配置
    BAIDU_API_KEY = 'V0ixU1bnNB36j41E8QxhsXiT'
    BAIDU_SECRET_KEY = 'QPOvVdYmTjws7vFAZUfRtYSfcMi07Onh'
    BAIDU_ACCESS_TOKEN = None

    # 百度API URLs
    BAIDU_TOKEN_URL = 'https://aip.baidubce.com/oauth/2.0/token'
    BAIDU_FACE_DETECT_URL = 'https://aip.baidubce.com/rest/2.0/face/v3/detect'
    BAIDU_FACE_SEARCH_URL = 'https://aip.baidubce.com/rest/2.0/face/v3/search'
    BAIDU_FACE_ADD_URL = 'https://aip.baidubce.com/rest/2.0/face/v3/faceset/user/add'
    BAIDU_LIVENESS_URL = 'https://aip.baidubce.com/rest/2.0/face/v3/faceverify'

    # 活体检测配置
    LIVENESS_THRESHOLD = 0.4  # 活体检测阈值

    # dlib活体检测权重配置
    DLIB_WEIGHTS = {
        'blink': 0.4,    # 眨眼检测权重
        'mouth': 0.4,    # 张嘴检测权重
        'pose': 0.2      # 头部姿态权重
    }

    # 摄像头配置
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480

    # 文件上传配置
    UPLOAD_FOLDER = 'static/uploads/'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB

    @staticmethod
    def init_directories():
        """初始化必要的目录"""
        directories = [
            Config.FACE_DB_PATH,
            Config.UPLOAD_FOLDER,
            os.path.dirname(Config.DATABASE_PATH)
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)


# 系统支持的活体检测方法
LIVENESS_METHODS = {
    'baidu_only': {
        'name': '百度API检测',
        'description': '使用百度云活体检测API',
        'key': 'baidu_only'
    },
    'dlib_only': {
        'name': 'dlib本地检测',
        'description': '眨眼40%+张嘴40%+姿态20%',
        'key': 'dlib_only'
    },
    'combined': {
        'name': '综合检测',
        'description': '百度API 50% + dlib检测 50%',
        'key': 'combined'
    }
}