new Vue({
    el: '#app',
    data() {
        return {
            // 基础配置
            apiBaseUrl: 'http://localhost:5000/api',
            activeTab: 'attendance',

            // 创建API实例
            api: null,

            // 摄像头相关
            cameraActive: false,
            cameraStream: null,
            processing: false,
            cameraConfig: {
                width: 640,
                height: 480
            },

            // 学生管理摄像头相关
            studentCameraActive: false,
            studentCameraStream: null,
            capturedPhotoPreview: null,

            // 系统状态
            systemStatus: {
                face_database_count: 0,
                baidu_api_available: false,
                dlib_available: false,
                liveness_threshold: 0.7
            },

            // 活体检测方法
            livenessMethods: [],
            selectedLivenessMethod: 'combined',

            // 考勤结果
            lastAttendanceResult: null,
            attendanceSummary: {
                total_students: 0,
                total_records: 0,
                avg_confidence: 0,
                avg_liveness_score: 0
            },

            // 学生管理
            students: [],
            studentSearchKeyword: '',
            loadingStudents: false,
            addingStudent: false,
            newStudent: {
                student_id: '',
                name: '',
                class_name: '',
                image: null
            },
            studentRules: {
                student_id: [
                    { required: true, message: '请输入学号', trigger: 'blur' },
                    { validator: this.validateStudentId, trigger: 'blur' }
                ],
                name: [
                    { required: true, message: '请输入姓名', trigger: 'blur' },
                    { validator: this.validateName, trigger: 'blur' }
                ],
                class_name: [
                    { required: true, message: '请输入班级', trigger: 'blur' },
                    { validator: this.validateClassName, trigger: 'blur' }
                ]
            },

            // 学生详情对话框
            studentDetailDialogVisible: false,
            selectedStudent: null,

            // 考勤记录
            attendanceRecords: [],
            loadingRecords: false,
            selectedDate: null,

            // 方法对比数据
            methodComparisonData: [
                {
                    method: '百度API',
                    accuracy: '★★★★★',
                    speed: '★★★★☆',
                    network: '需要',
                    features: '专业API，准确度高，支持多种攻击检测'
                },
                {
                    method: 'dlib本地',
                    accuracy: '★★★★☆',
                    speed: '★★★★★',
                    network: '不需要',
                    features: '本地运行，隐私保护，实时检测'
                },
                {
                    method: '综合检测',
                    accuracy: '★★★★★',
                    speed: '★★★☆☆',
                    network: '需要',
                    features: '结合两种方法优势，准确度最高'
                }
            ]
        }
    },

    computed: {
        // 过滤后的学生列表
        filteredStudents() {
            console.log('=== 计算filteredStudents ===');
            console.log('students原始数据:', this.students);
            console.log('studentSearchKeyword:', this.studentSearchKeyword);

            // 如果没有学生数据，返回空数组
            if (!this.students || !Array.isArray(this.students) || this.students.length === 0) {
                console.log('没有学生数据，返回空数组');
                return [];
            }

            // 如果没有搜索关键词，返回所有学生
            const keyword = this.studentSearchKeyword;
            if (!keyword || keyword.trim() === '') {
                console.log('无搜索条件，返回全部学生:', this.students.length, '个');
                return this.students;
            }

            // 进行搜索过滤
            const searchTerm = keyword.trim().toLowerCase();
            console.log('搜索关键词:', searchTerm);

            const filtered = this.students.filter((student) => {
                if (!student) return false;

                const studentId = String(student.student_id || '').toLowerCase();
                const name = String(student.name || '').toLowerCase();
                const className = String(student.class_name || '').toLowerCase();

                return studentId.includes(searchTerm) ||
                       name.includes(searchTerm) ||
                       className.includes(searchTerm);
            });

            console.log('过滤结果:', filtered.length, '个学生');
            return filtered;
        }
    },

    watch: {
        // 监听搜索关键词变化
        studentSearchKeyword: {
            handler(newVal) {
                console.log('搜索关键词变化:', newVal);
            },
            immediate: true
        }
    },

    mounted() {
        // 确保搜索关键词为空
        this.studentSearchKeyword = '';
        this.initializeSystem();
        this.loadUserPreferences();
    },

    beforeDestroy() {
        this.stopCamera();
        this.stopStudentCamera();
        this.saveUserPreferences();
    },

    methods: {
        // === 系统初始化 ===
        async initializeSystem() {
            try {
                // 初始化API实例
                this.api = ApiUtils.createInstance(this.apiBaseUrl);

                // 检查摄像头支持
                if (!CameraUtils.isSupported()) {
                    this.$message.error('当前浏览器不支持摄像头功能');
                }

                // 加载系统数据
                await Promise.all([
                    this.loadSystemStatus(),
                    this.loadLivenessMethods(),
                    this.loadStudents(),
                    this.loadAttendanceRecords(),
                    this.loadAttendanceSummary()
                ]);

                this.$message.success('系统初始化完成');
            } catch (error) {
                console.error('系统初始化失败:', error);
                this.$message.error('系统初始化失败，请检查网络连接');
            }
        },

        // 加载系统状态
        async loadSystemStatus() {
            try {
                const response = await this.api.get('/system_status');
                this.systemStatus = response.data;
                console.log('系统状态:', this.systemStatus);
            } catch (error) {
                console.error('加载系统状态失败:', error);
                this.$message.error('加载系统状态失败');
            }
        },

        // 加载活体检测方法
        async loadLivenessMethods() {
            try {
                const response = await this.api.get('/liveness_methods');
                this.livenessMethods = response.data.methods;
                this.selectedLivenessMethod = response.data.default_method;
                console.log('活体检测方法加载成功:', this.livenessMethods);

                // 如果没有加载到方法，使用默认方法
                if (!this.livenessMethods || this.livenessMethods.length === 0) {
                    this.livenessMethods = [
                        {
                            key: 'baidu_only',
                            name: '百度API检测',
                            description: '使用百度云活体检测API'
                        },
                        {
                            key: 'dlib_only',
                            name: 'dlib本地检测',
                            description: '眨眼40%+张嘴40%+姿态20%'
                        },
                        {
                            key: 'combined',
                            name: '综合检测',
                            description: '百度API 50% + dlib检测 50%'
                        }
                    ];
                    this.selectedLivenessMethod = 'combined';
                    console.log('使用默认活体检测方法');
                }
            } catch (error) {
                console.error('加载检测方法失败:', error);
                // 使用默认方法作为后备
                this.livenessMethods = [
                    {
                        key: 'baidu_only',
                        name: '百度API检测',
                        description: '使用百度云活体检测API'
                    },
                    {
                        key: 'dlib_only',
                        name: 'dlib本地检测',
                        description: '眨眼40%+张嘴40%+姿态20%'
                    },
                    {
                        key: 'combined',
                        name: '综合检测',
                        description: '百度API 50% + dlib检测 50%'
                    }
                ];
                this.selectedLivenessMethod = 'combined';
                this.$message.warning('使用默认检测方法，请检查后端连接');
            }
        },

        // === 摄像头控制 ===
        async startCamera() {
            try {
                // 检查权限
                const hasPermission = await CameraUtils.checkPermission();
                if (!hasPermission) {
                    this.$message.error('摄像头权限被拒绝，请在浏览器设置中允许摄像头访问');
                    return;
                }

                const constraints = {
                    video: {
                        width: this.cameraConfig.width,
                        height: this.cameraConfig.height,
                        facingMode: 'user'
                    }
                };

                this.cameraStream = await navigator.mediaDevices.getUserMedia(constraints);

                // 根据当前页面选择对应的video元素
                const videoElement = this.activeTab === 'students' ? this.$refs.studentVideo : this.$refs.video;
                if (videoElement) {
                    videoElement.srcObject = this.cameraStream;
                }

                this.cameraActive = true;
                this.$message.success('摄像头启动成功');
            } catch (error) {
                console.error('摄像头启动失败:', error);
                this.$message.error('摄像头启动失败，请检查设备和权限设置');
            }
        },

        stopCamera() {
            if (this.cameraStream) {
                this.cameraStream.getTracks().forEach(track => track.stop());
                this.cameraStream = null;
            }
            this.cameraActive = false;
            this.$message.info('摄像头已关闭');
        },

        // === 学生管理摄像头控制 ===
        async startStudentCamera() {
            try {
                console.log('启动学生摄像头...');

                // 检查权限
                const hasPermission = await CameraUtils.checkPermission();
                if (!hasPermission) {
                    this.$message.error('摄像头权限被拒绝，请在浏览器设置中允许摄像头访问');
                    return false;
                }

                const constraints = {
                    video: {
                        width: 320,
                        height: 240,
                        facingMode: 'user'
                    }
                };

                this.studentCameraStream = await navigator.mediaDevices.getUserMedia(constraints);

                // 设置学生摄像头视频流
                const videoElement = this.$refs.studentVideo;
                if (videoElement) {
                    videoElement.srcObject = this.studentCameraStream;
                }

                this.studentCameraActive = true;
                this.$message.success('学生摄像头启动成功');
                return true;
            } catch (error) {
                console.error('学生摄像头启动失败:', error);
                this.$message.error('摄像头启动失败，请检查设备和权限设置');
                return false;
            }
        },

        stopStudentCamera() {
            if (this.studentCameraStream) {
                this.studentCameraStream.getTracks().forEach(track => track.stop());
                this.studentCameraStream = null;
            }
            this.studentCameraActive = false;
            console.log('学生摄像头已关闭');
        },

        async toggleStudentCamera() {
            if (this.studentCameraActive) {
                this.stopStudentCamera();
                this.$message.info('摄像头已关闭');
            } else {
                await this.startStudentCamera();
            }
        },

        // 拍摄照片
        async capturePhoto(isStudentMode = false) {
            // 根据模式选择对应的摄像头和元素
            const cameraActive = isStudentMode ? this.studentCameraActive : this.cameraActive;

            if (!cameraActive) {
                this.$message.error('请先启动摄像头');
                return null;
            }

            try {
                // 根据模式选择对应的canvas和video元素
                const canvas = isStudentMode ? this.$refs.studentCanvas : this.$refs.canvas;
                const video = isStudentMode ? this.$refs.studentVideo : this.$refs.video;

                if (!canvas || !video) {
                    this.$message.error('摄像头元素未找到');
                    return null;
                }

                const context = canvas.getContext('2d');
                context.drawImage(video, 0, 0, canvas.width, canvas.height);

                // 获取base64数据并压缩
                const originalBase64 = canvas.toDataURL('image/jpeg', 0.8);
                const compressedBase64 = await FileUtils.compressImage(originalBase64, 0.8, 800);

                return compressedBase64;
            } catch (error) {
                console.error('拍摄照片失败:', error);
                this.$message.error('拍摄照片失败');
                return null;
            }
        },

        // === 考勤打卡 ===
        async captureAndAttendance() {
            this.processing = true;

            try {
                const imageData = await this.capturePhoto(false); // 考勤模式
                if (!imageData) {
                    return;
                }

                const requestData = {
                    image: imageData,
                    liveness_method: this.selectedLivenessMethod
                };

                console.log('开始考勤打卡，检测方法:', this.selectedLivenessMethod);

                const response = await this.api.post('/attendance', requestData);
                this.lastAttendanceResult = response.data;

                if (response.data.success) {
                    this.$message.success('考勤打卡成功！');
                    // 刷新考勤记录和统计
                    await Promise.all([
                        this.loadAttendanceRecords(),
                        this.loadAttendanceSummary()
                    ]);
                } else {
                    this.$message.error(response.data.error || '考勤失败');
                }

            } catch (error) {
                console.error('考勤处理失败:', error);
                this.$message.error('考勤处理失败，请重试');
                this.lastAttendanceResult = {
                    success: false,
                    error: '网络请求失败'
                };
            } finally {
                this.processing = false;
            }
        },

        // 清除最后一次检测结果
        clearLastResult() {
            this.lastAttendanceResult = null;
        },

        // === 学生管理 ===
        async loadStudents() {
            this.loadingStudents = true;
            try {
                console.log('开始加载学生列表...');
                const response = await this.api.get('/students');
                console.log('学生列表API完整响应:', response);
                console.log('学生列表API响应数据:', response.data);

                // 检查响应结构
                if (response.data && response.data.success && response.data.students) {
                    this.students = response.data.students;
                    console.log('成功设置学生数据:', this.students);
                    console.log('Vue data中的students:', this.$data.students);

                    // 强制触发响应式更新
                    this.$forceUpdate();
                    console.log('已强制更新视图');

                    if (this.students.length > 0) {
                        console.log('第一个学生数据示例:', this.students[0]);
                    }
                } else {
                    console.error('API响应格式不正确:', response.data);
                    this.$message.error('学生数据格式不正确');
                    this.students = [];
                }
            } catch (error) {
                console.error('加载学生列表请求失败:', error);
                this.$message.error('加载学生列表失败，请检查网络连接');
                this.students = [];
            } finally {
                this.loadingStudents = false;
            }
        },

        // 清除搜索
        clearSearch() {
            this.studentSearchKeyword = '';
            console.log('已清除搜索，重置为显示全部学生');
        },

        // 调试学生数据
        async debugStudentData() {
            console.log('=== 学生数据调试信息 ===');
            console.log('前端数据:');
            console.log('- 原始学生数组:', this.students);
            console.log('- 学生数组长度:', this.students.length);
            console.log('- 过滤后学生数组:', this.filteredStudents);
            console.log('- 过滤后数组长度:', this.filteredStudents.length);
            console.log('- 搜索关键词:', this.studentSearchKeyword);

            if (this.students.length > 0) {
                console.log('- 第一个学生数据结构:', this.students[0]);
                console.log('- 学生数据键名:', Object.keys(this.students[0]));
            }

            try {
                // 调用后端调试API
                console.log('后端数据:');
                const response = await this.api.get('/debug/students');
                console.log('- 后端调试信息:', response.data);

                // 重新加载学生数据
                await this.loadStudents();

                // 显示调试信息
                const debugInfo = {
                    前端信息: {
                        原始数据数量: this.students.length,
                        过滤后数量: this.filteredStudents.length,
                        搜索关键词: this.studentSearchKeyword || '无',
                        数据示例: this.students[0] || '无数据'
                    },
                    后端信息: response.data.debug_info
                };

                this.$alert(
                    `<pre style="max-height: 400px; overflow-y: auto;">${JSON.stringify(debugInfo, null, 2)}</pre>`,
                    '学生数据调试信息',
                    {
                        dangerouslyUseHTMLString: true,
                        type: 'info'
                    }
                );

            } catch (error) {
                console.error('调试请求失败:', error);

                // 显示前端调试信息
                const debugInfo = {
                    前端信息: {
                        原始数据数量: this.students.length,
                        过滤后数量: this.filteredStudents.length,
                        搜索关键词: this.studentSearchKeyword || '无',
                        数据示例: this.students[0] || '无数据'
                    },
                    后端信息: '调用失败: ' + error.message
                };

                this.$alert(
                    `<pre>${JSON.stringify(debugInfo, null, 2)}</pre>`,
                    '学生数据调试信息',
                    {
                        dangerouslyUseHTMLString: true,
                        type: 'warning'
                    }
                );
            }
        },

        async captureStudentPhoto() {
            try {
                console.log('开始拍摄学生照片...');

                // 检查摄像头状态
                if (!this.studentCameraActive) {
                    this.$message.error('请先启动摄像头');
                    return;
                }

                console.log('学生摄像头已就绪，开始拍摄...');

                const imageData = await this.capturePhoto(true); // 学生模式
                if (imageData) {
                    console.log('照片拍摄成功，数据长度:', imageData.length);

                    // 验证图像数据格式
                    if (!imageData.startsWith('data:image/')) {
                        this.$message.error('图像格式不正确');
                        return;
                    }

                    // 设置预览图像
                    this.capturedPhotoPreview = imageData;
                    this.$message.success('照片拍摄成功！请确认照片质量');

                } else {
                    this.$message.error('照片拍摄失败，请重试');
                }
            } catch (error) {
                console.error('拍摄学生照片失败:', error);
                this.$message.error('拍摄照片失败，请检查摄像头设备');
            }
        },

        // 确认使用拍摄的照片
        confirmPhoto() {
            if (this.capturedPhotoPreview) {
                this.newStudent.image = this.capturedPhotoPreview;
                this.capturedPhotoPreview = null;
                this.$message.success('照片已确认！');

                // 显示成功信息
                this.$notify({
                    title: '照片确认成功',
                    message: '人脸照片已确认，可以提交表单了',
                    type: 'success',
                    duration: 3000
                });
            }
        },

        // 重新拍摄照片
        retakePhoto() {
            this.capturedPhotoPreview = null;
            this.$message.info('已清除照片，请重新拍摄');
        },

        async addStudent() {
            try {
                // 表单验证
                const valid = await this.$refs.studentForm.validate().catch(() => false);
                if (!valid) {
                    this.$message.error('请检查表单输入');
                    return false;
                }

                if (!this.newStudent.image) {
                    this.$message.error('请先拍摄人脸照片');
                    return false;
                }

                // 数据验证
                if (!this.newStudent.student_id.trim()) {
                    this.$message.error('学号不能为空');
                    return;
                }

                if (!this.newStudent.name.trim()) {
                    this.$message.error('姓名不能为空');
                    return;
                }

                if (!this.newStudent.class_name.trim()) {
                    this.$message.error('班级不能为空');
                    return;
                }

                this.addingStudent = true;

                // 准备请求数据
                const studentData = {
                    student_id: this.newStudent.student_id.trim(),
                    name: this.newStudent.name.trim(),
                    class_name: this.newStudent.class_name.trim(),
                    image: this.newStudent.image
                };

                console.log('开始添加学生:', studentData.name, '学号:', studentData.student_id);

                const response = await this.api.post('/add_student', studentData);

                console.log('服务器响应:', response.data);

                if (response.data.success) {
                    this.$message.success('学生添加成功！');

                    // 显示详细信息
                    this.$notify({
                        title: '添加成功',
                        message: `学生 ${studentData.name} (${studentData.student_id}) 已成功添加到系统`,
                        type: 'success',
                        duration: 5000
                    });

                    this.resetStudentForm();

                    // 刷新相关数据
                    await Promise.all([
                        this.loadStudents(),
                        this.loadSystemStatus()  // 更新人脸库数量
                    ]);
                } else {
                    this.$message.error(response.data.error || '添加失败');
                }

            } catch (error) {
                console.error('添加学生请求失败:', error);

                let errorMessage = '添加学生失败';

                if (error.response) {
                    // 服务器返回错误
                    const serverError = error.response.data;
                    if (serverError && serverError.error) {
                        errorMessage = serverError.error;
                    } else {
                        errorMessage = `服务器错误 (${error.response.status})`;
                    }
                    console.error('服务器错误详情:', serverError);
                } else if (error.request) {
                    // 网络请求失败
                    errorMessage = '网络请求失败，请检查网络连接';
                    console.error('网络请求失败:', error.request);
                } else {
                    // 其他错误
                    errorMessage = `请求配置错误: ${error.message}`;
                    console.error('请求配置错误:', error.message);
                }

                this.$message.error(errorMessage);

                // 显示详细错误信息
                this.$notify({
                    title: '添加失败',
                    message: errorMessage,
                    type: 'error',
                    duration: 8000
                });

            } finally {
                this.addingStudent = false;
            }
        },

        resetStudentForm() {
            this.newStudent = {
                student_id: '',
                name: '',
                class_name: '',
                image: null
            };

            // 重置照片相关状态
            this.capturedPhotoPreview = null;

            // 关闭学生摄像头
            this.stopStudentCamera();

            if (this.$refs.studentForm) {
                this.$refs.studentForm.resetFields();
            }
            console.log('学生表单已重置');
        },

        // 调试用：测试添加学生功能
        async testAddStudent() {
            console.log('=== 开始调试学生添加功能 ===');

            try {
                // 1. 检查本地数据
                console.log('1. 本地数据检查:');
                console.log('- 学生数据:', this.newStudent);
                console.log('- 摄像头状态:', this.cameraActive);
                console.log('- 图像数据存在:', !!this.newStudent.image);

                if (this.newStudent.image) {
                    console.log('- 图像数据长度:', this.newStudent.image.length);
                    console.log('- 图像数据格式:', this.newStudent.image.substring(0, 50) + '...');
                }

                // 2. 测试API连接
                console.log('2. API连接测试:');
                try {
                    const healthResponse = await this.api.get('/health');
                    console.log('- 健康检查通过:', healthResponse.data);
                } catch (error) {
                    console.error('- API连接失败:', error);
                }

                // 3. 发送调试请求
                console.log('3. 调试分析:');
                try {
                    const debugResponse = await this.api.post('/debug/student_add', this.newStudent);
                    console.log('- 调试分析结果:', debugResponse.data);

                    // 显示调试信息
                    this.$alert(
                        `<pre>${JSON.stringify(debugResponse.data.debug_info, null, 2)}</pre>`,
                        '调试信息',
                        {
                            dangerouslyUseHTMLString: true,
                            type: 'info'
                        }
                    );
                } catch (error) {
                    console.error('- 调试请求失败:', error);
                }

                console.log('=== 调试完成 ===');

            } catch (error) {
                console.error('调试过程中发生错误:', error);
                this.$message.error('调试失败: ' + error.message);
            }
        },

        viewStudent(student) {
            this.selectedStudent = student;
            this.studentDetailDialogVisible = true;
        },

        deleteStudent(student) {
            this.$confirm(`确定要删除学生 ${student.name} (${student.student_id}) 吗？`, '删除确认', {
                confirmButtonText: '确定删除',
                cancelButtonText: '取消',
                type: 'warning',
                dangerouslyUseHTMLString: true
            }).then(async () => {
                try {
                    const response = await this.api.delete(`/delete_student/${student.student_id}`);

                    if (response.data.success) {
                        this.$message.success('学生删除成功');
                        await Promise.all([
                            this.loadStudents(),
                            this.loadSystemStatus()  // 更新人脸库数量
                        ]);
                    } else {
                        this.$message.error(response.data.error || '删除失败');
                    }
                } catch (error) {
                    console.error('删除学生失败:', error);
                    this.$message.error('删除学生失败，请重试');
                }
            }).catch(() => {
                this.$message.info('已取消删除');
            });
        },

        // === 考勤记录 ===
        async loadAttendanceRecords() {
            this.loadingRecords = true;
            try {
                const params = this.selectedDate ? { date: this.selectedDate } : {};
                const response = await this.api.get('/attendance_records', { params });
                this.attendanceRecords = response.data.records || [];
                console.log('加载考勤记录:', this.attendanceRecords.length);
            } catch (error) {
                console.error('加载考勤记录失败:', error);
                this.$message.error('加载考勤记录失败');
            } finally {
                this.loadingRecords = false;
            }
        },

        async loadAttendanceSummary() {
            try {
                const params = this.selectedDate ? { date: this.selectedDate } : {};
                const response = await this.api.get('/attendance_summary', { params });
                this.attendanceSummary = response.data.summary || {};
                console.log('考勤统计:', this.attendanceSummary);
            } catch (error) {
                console.error('加载考勤统计失败:', error);
            }
        },

        // 导出考勤记录
        exportRecords() {
            if (this.attendanceRecords.length === 0) {
                this.$message.warning('暂无数据可导出');
                return;
            }

            try {
                // 处理导出数据
                const exportData = this.attendanceRecords.map(record => ({
                    '学号': record.student_id,
                    '姓名': record.student_name,
                    '打卡时间': TimeUtils.formatTime(record.check_time),
                    '检测方法': this.getMethodDisplayName(record.detection_method),
                    '识别置信度': FormatUtils.formatPercentage(record.confidence),
                    '活体分数': FormatUtils.formatPercentage(record.liveness_score),
                    '状态': record.status
                }));

                const filename = `考勤记录_${this.selectedDate || TimeUtils.getTodayString()}.csv`;
                FileUtils.exportToCSV(exportData, filename);
                this.$message.success('导出成功');
            } catch (error) {
                console.error('导出失败:', error);
                this.$message.error('导出失败');
            }
        },

        // === 工具函数 ===
        getCurrentMethodInfo() {
            const method = this.livenessMethods.find(m => m.key === this.selectedLivenessMethod);
            return method || { name: '未知方法', description: '' };
        },

        onLivenessMethodChange(value) {
            console.log('活体检测方法变更:', value);
            const method = this.livenessMethods.find(m => m.key === value);
            if (method) {
                this.$message({
                    message: `已切换到 ${method.name}`,
                    type: 'success',
                    duration: 2000,
                    showClose: true
                });

                // 显示方法说明
                this.$notify({
                    title: '检测方法说明',
                    message: method.description,
                    type: 'info',
                    duration: 4000
                });

                // 保存用户偏好
                this.saveUserPreferences();
            }
        },

        // 快速切换检测方法
        quickSwitchMethod(methodKey) {
            this.selectedLivenessMethod = methodKey;
            this.onLivenessMethodChange(methodKey);
        },

        getProgressColor(percentage) {
            if (percentage >= 90) return '#67C23A';
            if (percentage >= 80) return '#E6A23C';
            if (percentage >= 70) return '#F56C6C';
            return '#F56C6C';
        },

        getMethodTagType(method) {
            switch (method) {
                case 'baidu_only': return 'primary';
                case 'dlib_only': return 'success';
                case 'combined': return 'warning';
                default: return 'info';
            }
        },

        getMethodDisplayName(method) {
            const methodInfo = this.livenessMethods.find(m => m.key === method);
            return methodInfo ? methodInfo.name : method;
        },

        formatTime(timestamp) {
            return TimeUtils.formatTime(timestamp);
        },

        formatTableTime(row, column, cellValue) {
            return TimeUtils.formatTime(cellValue);
        },

        // === 表单验证 ===
        validateStudentId(rule, value, callback) {
            if (!ValidationUtils.validateStudentId(value)) {
                callback(new Error('学号格式不正确，应为4-20位字母数字组合'));
            } else {
                callback();
            }
        },

        validateName(rule, value, callback) {
            if (!ValidationUtils.validateName(value)) {
                callback(new Error('姓名格式不正确，应为2-50个字符'));
            } else {
                callback();
            }
        },

        validateClassName(rule, value, callback) {
            if (!ValidationUtils.validateClassName(value)) {
                callback(new Error('班级名称格式不正确，应为2-30个字符'));
            } else {
                callback();
            }
        },

        // === 用户偏好设置 ===
        loadUserPreferences() {
            try {
                const preferences = StorageUtils.get('user_preferences', {});

                // 恢复活体检测方法选择
                if (preferences.selectedLivenessMethod) {
                    this.selectedLivenessMethod = preferences.selectedLivenessMethod;
                }

                // 恢复其他偏好设置
                if (preferences.activeTab) {
                    this.activeTab = preferences.activeTab;
                }

                console.log('加载用户偏好设置:', preferences);
            } catch (error) {
                console.error('加载用户偏好设置失败:', error);
            }
        },

        saveUserPreferences() {
            try {
                const preferences = {
                    selectedLivenessMethod: this.selectedLivenessMethod,
                    activeTab: this.activeTab,
                    lastSaveTime: new Date().toISOString()
                };

                StorageUtils.set('user_preferences', preferences);
                console.log('保存用户偏好设置:', preferences);
            } catch (error) {
                console.error('保存用户偏好设置失败:', error);
            }
        },

        // === 选项卡切换 ===
        handleTabClick(tab) {
            console.log('切换到选项卡:', tab.name);

            // 保存当前选项卡
            this.activeTab = tab.name;

            // 根据选项卡加载对应数据
            switch (tab.name) {
                case 'students':
                    this.loadStudents();
                    break;
                case 'records':
                    this.loadAttendanceRecords();
                    break;
                case 'attendance':
                    this.loadAttendanceSummary();
                    // 如果从学生管理页面切换回来，关闭学生摄像头
                    if (this.studentCameraActive) {
                        this.stopStudentCamera();
                    }
                    break;
            }

            // 保存用户偏好
            this.saveUserPreferences();
        }
    }
});