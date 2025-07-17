import sqlite3
import os
import json
import numpy as np


class DatabaseManager:
    def __init__(self, db_path='../database/attendance.db'):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """初始化数据库表"""
        # 确保数据库目录存在
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 学生信息表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id VARCHAR(20) UNIQUE NOT NULL,
                name VARCHAR(50) NOT NULL,
                class_name VARCHAR(30) NOT NULL,
                face_encoding TEXT,  -- 存储人脸编码的JSON字符串
                photo_path VARCHAR(200),
                face_quality_score FLOAT DEFAULT 0.0,  -- 人脸质量评分
                is_active BOOLEAN DEFAULT 1,  -- 是否激活
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # 考勤记录表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id VARCHAR(20) NOT NULL,
                student_name VARCHAR(50) NOT NULL,
                check_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                detection_method VARCHAR(20) NOT NULL,
                confidence FLOAT,
                liveness_score FLOAT,
                status VARCHAR(20) DEFAULT 'present',
                photo_path VARCHAR(200),  -- 考勤时的照片
                FOREIGN KEY (student_id) REFERENCES students (student_id)
            )
        ''')

        # 系统配置表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_config (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                config_key VARCHAR(50) UNIQUE NOT NULL,
                config_value TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        conn.commit()
        conn.close()
        print("数据库初始化完成!")

    def face_encoding_to_string(self, face_encoding):
        """将人脸编码转换为字符串存储"""
        if face_encoding is not None:
            return json.dumps(face_encoding.tolist())
        return None

    def string_to_face_encoding(self, encoding_string):
        """将字符串转换为人脸编码"""
        if encoding_string:
            return np.array(json.loads(encoding_string))
        return None

    def add_student(self, student_id, name, class_name, face_encoding=None,
                    photo_path=None, face_quality_score=0.0):
        """添加学生信息（包括人脸编码）"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # 检查学生ID是否已存在
            cursor.execute('SELECT student_id FROM students WHERE student_id = ?', (student_id,))
            if cursor.fetchone():
                return False, "学生ID已存在"

            # 将人脸编码转换为字符串
            encoding_str = self.face_encoding_to_string(face_encoding)

            cursor.execute('''
                INSERT INTO students (student_id, name, class_name, face_encoding, 
                                    photo_path, face_quality_score)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (student_id, name, class_name, encoding_str, photo_path, face_quality_score))

            conn.commit()
            print(f"成功添加学生: {name} ({student_id})")
            return True, "学生添加成功"
        except sqlite3.IntegrityError as e:
            print(f"添加学生失败: {e}")
            return False, "学生ID已存在"
        except Exception as e:
            print(f"添加学生时发生错误: {e}")
            return False, f"添加失败: {str(e)}"
        finally:
            conn.close()

    def delete_student(self, student_id):
        """删除学生（物理删除）"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # 先检查学生是否存在
            cursor.execute('SELECT name FROM students WHERE student_id = ?', (student_id,))
            student = cursor.fetchone()
            if not student:
                return False, "学生不存在"

            # 删除学生记录
            cursor.execute('DELETE FROM students WHERE student_id = ?', (student_id,))

            # 可选：同时删除该学生的考勤记录
            # cursor.execute('DELETE FROM attendance_records WHERE student_id = ?', (student_id,))

            conn.commit()

            if cursor.rowcount > 0:
                print(f"成功删除学生: {student[0]} ({student_id})")
                return True, "学生删除成功"
            else:
                return False, "删除失败"

        except Exception as e:
            print(f"删除学生时发生错误: {e}")
            return False, f"删除失败: {str(e)}"
        finally:
            conn.close()

    def deactivate_student(self, student_id):
        """停用学生（软删除）"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute('''
                UPDATE students 
                SET is_active = 0, updated_at = CURRENT_TIMESTAMP
                WHERE student_id = ?
            ''', (student_id,))

            conn.commit()
            success = cursor.rowcount > 0

            if success:
                print(f"成功停用学生: {student_id}")

            return success, "学生停用成功" if success else "学生不存在"
        except Exception as e:
            print(f"停用学生时发生错误: {e}")
            return False, f"停用失败: {str(e)}"
        finally:
            conn.close()

    def update_student_face_encoding(self, student_id, face_encoding, face_quality_score=None):
        """更新学生的人脸编码"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            encoding_str = self.face_encoding_to_string(face_encoding)

            if face_quality_score is not None:
                cursor.execute('''
                    UPDATE students 
                    SET face_encoding = ?, face_quality_score = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE student_id = ?
                ''', (encoding_str, face_quality_score, student_id))
            else:
                cursor.execute('''
                    UPDATE students 
                    SET face_encoding = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE student_id = ?
                ''', (encoding_str, student_id))

            conn.commit()
            success = cursor.rowcount > 0

            if success:
                print(f"成功更新学生人脸编码: {student_id}")

            return success
        except Exception as e:
            print(f"更新人脸编码失败: {e}")
            return False
        finally:
            conn.close()

    def get_student(self, student_id):
        """获取学生信息"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM students WHERE student_id = ? AND is_active = 1', (student_id,))
        result = cursor.fetchone()
        conn.close()

        return result

    def get_all_students(self):
        """获取所有激活的学生信息"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM students WHERE is_active = 1 ORDER BY class_name, name')
        results = cursor.fetchall()
        conn.close()

        return results

    def get_all_face_encodings(self):
        """获取所有学生的人脸编码"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT student_id, name, face_encoding 
            FROM students 
            WHERE is_active = 1 AND face_encoding IS NOT NULL
        ''')
        results = cursor.fetchall()
        conn.close()

        face_data = []
        for student_id, name, encoding_str in results:
            if encoding_str:
                try:
                    encoding = self.string_to_face_encoding(encoding_str)
                    if encoding is not None:
                        face_data.append({
                            'student_id': student_id,
                            'name': name,
                            'encoding': encoding
                        })
                except Exception as e:
                    print(f"解析人脸编码失败 {student_id}: {e}")

        return face_data

    def add_attendance_record(self, student_id, student_name, detection_method,
                              confidence=0.0, liveness_score=0.0, photo_path=None):
        """添加考勤记录"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute('''
                INSERT INTO attendance_records 
                (student_id, student_name, detection_method, confidence, liveness_score, photo_path)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (student_id, student_name, detection_method, confidence, liveness_score, photo_path))

            conn.commit()
            print(f"添加考勤记录成功: {student_name} ({student_id})")
            return True
        except Exception as e:
            print(f"添加考勤记录失败: {e}")
            return False
        finally:
            conn.close()

    def get_today_attendance(self):
        """获取今日考勤记录"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT * FROM attendance_records 
            WHERE DATE(check_time) = DATE('now')
            ORDER BY check_time DESC
        ''')

        results = cursor.fetchall()
        conn.close()

        return results

    def get_attendance_by_date(self, date):
        """按日期获取考勤记录"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT * FROM attendance_records 
            WHERE DATE(check_time) = ?
            ORDER BY check_time DESC
        ''', (date,))

        results = cursor.fetchall()
        conn.close()

        return results

    def get_student_attendance_stats(self, student_id, days=30):
        """获取学生考勤统计"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT COUNT(*) as attendance_count
            FROM attendance_records 
            WHERE student_id = ? AND check_time >= datetime('now', '-{} days')
        '''.format(days), (student_id,))

        result = cursor.fetchone()
        conn.close()

        return result[0] if result else 0

    def get_attendance_summary(self, date=None):
        """获取考勤汇总统计"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if date:
            cursor.execute('''
                SELECT 
                    COUNT(DISTINCT student_id) as total_students,
                    COUNT(*) as total_records,
                    AVG(confidence) as avg_confidence,
                    AVG(liveness_score) as avg_liveness_score
                FROM attendance_records 
                WHERE DATE(check_time) = ?
            ''', (date,))
        else:
            cursor.execute('''
                SELECT 
                    COUNT(DISTINCT student_id) as total_students,
                    COUNT(*) as total_records,
                    AVG(confidence) as avg_confidence,
                    AVG(liveness_score) as avg_liveness_score
                FROM attendance_records 
                WHERE DATE(check_time) = DATE('now')
            ''')

        result = cursor.fetchone()
        conn.close()

        if result:
            return {
                'total_students': result[0] or 0,
                'total_records': result[1] or 0,
                'avg_confidence': result[2] or 0.0,
                'avg_liveness_score': result[3] or 0.0
            }
        else:
            return {
                'total_students': 0,
                'total_records': 0,
                'avg_confidence': 0.0,
                'avg_liveness_score': 0.0
            }