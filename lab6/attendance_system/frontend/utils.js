// 时间格式化工具
const TimeUtils = {
    /**
     * 格式化时间戳为中文时间字符串
     * @param {string|Date} timestamp - 时间戳或Date对象
     * @returns {string} 格式化后的时间字符串
     */
    formatTime(timestamp) {
        if (!timestamp) return '';
        const date = new Date(timestamp);

        // 检查日期是否有效
        if (isNaN(date.getTime())) return '';

        return date.toLocaleString('zh-CN', {
            year: 'numeric',
            month: '2-digit',
            day: '2-digit',
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit',
            hour12: false
        });
    },

    /**
     * 格式化为相对时间（如：2分钟前）
     * @param {string|Date} timestamp - 时间戳或Date对象
     * @returns {string} 相对时间字符串
     */
    formatRelativeTime(timestamp) {
        if (!timestamp) return '';
        const date = new Date(timestamp);
        const now = new Date();
        const diff = now.getTime() - date.getTime();

        const seconds = Math.floor(diff / 1000);
        const minutes = Math.floor(seconds / 60);
        const hours = Math.floor(minutes / 60);
        const days = Math.floor(hours / 24);

        if (days > 0) return `${days}天前`;
        if (hours > 0) return `${hours}小时前`;
        if (minutes > 0) return `${minutes}分钟前`;
        return '刚刚';
    },

    /**
     * 获取今天的日期字符串
     * @returns {string} YYYY-MM-DD格式的日期字符串
     */
    getTodayString() {
        const today = new Date();
        return today.toISOString().split('T')[0];
    }
};

// 数据验证工具
const ValidationUtils = {
    /**
     * 验证学号格式
     * @param {string} studentId - 学号
     * @returns {boolean} 是否有效
     */
    validateStudentId(studentId) {
        if (!studentId) return false;
        // 假设学号为4-20位数字或字母数字组合
        const pattern = /^[A-Za-z0-9]{4,20}$/;
        return pattern.test(studentId);
    },

    /**
     * 验证姓名格式
     * @param {string} name - 姓名
     * @returns {boolean} 是否有效
     */
    validateName(name) {
        if (!name) return false;
        // 2-50个字符，支持中文、英文、数字
        const pattern = /^[\u4e00-\u9fa5A-Za-z0-9\s]{2,50}$/;
        return pattern.test(name.trim());
    },

    /**
     * 验证班级名称格式
     * @param {string} className - 班级名称
     * @returns {boolean} 是否有效
     */
    validateClassName(className) {
        if (!className) return false;
        // 2-30个字符
        const pattern = /^[\u4e00-\u9fa5A-Za-z0-9\s]{2,30}$/;
        return pattern.test(className.trim());
    }
};

// 文件处理工具
const FileUtils = {
    /**
     * 将表格数据导出为CSV文件
     * @param {Array} data - 表格数据
     * @param {string} filename - 文件名
     */
    exportToCSV(data, filename = 'export.csv') {
        if (!data || data.length === 0) {
            console.warn('没有数据可导出');
            return;
        }

        // 获取表头
        const headers = Object.keys(data[0]);

        // 构建CSV内容
        let csvContent = '\uFEFF'; // 添加BOM以支持中文
        csvContent += headers.join(',') + '\n';

        data.forEach(row => {
            const values = headers.map(header => {
                const value = row[header];
                // 处理包含逗号或换行的值
                if (typeof value === 'string' && (value.includes(',') || value.includes('\n'))) {
                    return `"${value.replace(/"/g, '""')}"`;
                }
                return value || '';
            });
            csvContent += values.join(',') + '\n';
        });

        // 创建下载链接
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement('a');

        if (link.download !== undefined) {
            const url = URL.createObjectURL(blob);
            link.setAttribute('href', url);
            link.setAttribute('download', filename);
            link.style.visibility = 'hidden';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }
    },

    /**
     * 压缩图片
     * @param {string} base64 - base64图片数据
     * @param {number} quality - 压缩质量 (0-1)
     * @param {number} maxWidth - 最大宽度
     * @returns {Promise<string>} 压缩后的base64数据
     */
    compressImage(base64, quality = 0.8, maxWidth = 800) {
        return new Promise((resolve) => {
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            const img = new Image();

            img.onload = () => {
                // 计算新尺寸
                let { width, height } = img;
                if (width > maxWidth) {
                    height = (height * maxWidth) / width;
                    width = maxWidth;
                }

                canvas.width = width;
                canvas.height = height;

                // 绘制并压缩
                ctx.drawImage(img, 0, 0, width, height);
                const compressedBase64 = canvas.toDataURL('image/jpeg', quality);
                resolve(compressedBase64);
            };

            img.src = base64;
        });
    }
};

// 数据格式化工具
const FormatUtils = {
    /**
     * 格式化百分比
     * @param {number} value - 数值 (0-1)
     * @param {number} decimals - 小数位数
     * @returns {string} 格式化后的百分比字符串
     */
    formatPercentage(value, decimals = 1) {
        if (typeof value !== 'number' || isNaN(value)) return '0%';
        return (value * 100).toFixed(decimals) + '%';
    },

    /**
     * 格式化置信度分数
     * @param {number} score - 分数 (0-1)
     * @returns {object} 包含颜色和文本的对象
     */
    formatConfidenceScore(score) {
        if (typeof score !== 'number' || isNaN(score)) {
            return { color: '#909399', text: '0%', level: 'unknown' };
        }

        const percentage = Math.round(score * 100);

        if (percentage >= 90) {
            return { color: '#67C23A', text: `${percentage}%`, level: 'excellent' };
        } else if (percentage >= 80) {
            return { color: '#E6A23C', text: `${percentage}%`, level: 'good' };
        } else if (percentage >= 70) {
            return { color: '#F56C6C', text: `${percentage}%`, level: 'fair' };
        } else {
            return { color: '#F56C6C', text: `${percentage}%`, level: 'poor' };
        }
    }
};

// 摄像头工具
const CameraUtils = {
    /**
     * 检查浏览器是否支持摄像头
     * @returns {boolean} 是否支持
     */
    isSupported() {
        return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
    },

    /**
     * 获取可用的摄像头设备列表
     * @returns {Promise<Array>} 摄像头设备列表
     */
    async getDevices() {
        try {
            const devices = await navigator.mediaDevices.enumerateDevices();
            return devices.filter(device => device.kind === 'videoinput');
        } catch (error) {
            console.error('获取摄像头设备失败:', error);
            return [];
        }
    },

    /**
     * 检查摄像头权限
     * @returns {Promise<boolean>} 是否有权限
     */
    async checkPermission() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            stream.getTracks().forEach(track => track.stop());
            return true;
        } catch (error) {
            console.error('摄像头权限检查失败:', error);
            return false;
        }
    }
};

// 本地存储工具
const StorageUtils = {
    /**
     * 设置本地存储
     * @param {string} key - 键名
     * @param {any} value - 值
     */
    set(key, value) {
        try {
            const serializedValue = JSON.stringify(value);
            localStorage.setItem(key, serializedValue);
        } catch (error) {
            console.error('设置本地存储失败:', error);
        }
    },

    /**
     * 获取本地存储
     * @param {string} key - 键名
     * @param {any} defaultValue - 默认值
     * @returns {any} 存储的值
     */
    get(key, defaultValue = null) {
        try {
            const serializedValue = localStorage.getItem(key);
            if (serializedValue === null) return defaultValue;
            return JSON.parse(serializedValue);
        } catch (error) {
            console.error('获取本地存储失败:', error);
            return defaultValue;
        }
    },

    /**
     * 删除本地存储
     * @param {string} key - 键名
     */
    remove(key) {
        try {
            localStorage.removeItem(key);
        } catch (error) {
            console.error('删除本地存储失败:', error);
        }
    },

    /**
     * 清空本地存储
     */
    clear() {
        try {
            localStorage.clear();
        } catch (error) {
            console.error('清空本地存储失败:', error);
        }
    }
};

// 网络请求工具
const ApiUtils = {
    /**
     * 创建带有通用配置的axios实例
     * @param {string} baseURL - 基础URL
     * @returns {object} axios实例
     */
    createInstance(baseURL) {
        const instance = axios.create({
            baseURL,
            timeout: 30000,  // 30秒超时
            headers: {
                'Content-Type': 'application/json'
            }
        });

        // 请求拦截器
        instance.interceptors.request.use(
            config => {
                console.log(`API请求: ${config.method?.toUpperCase()} ${config.url}`);
                return config;
            },
            error => {
                console.error('API请求错误:', error);
                return Promise.reject(error);
            }
        );

        // 响应拦截器
        instance.interceptors.response.use(
            response => {
                console.log(`API响应: ${response.config.url} - ${response.status}`);
                return response;
            },
            error => {
                console.error('API响应错误:', error);
                if (error.response) {
                    // 服务器返回错误状态码
                    const { status, data } = error.response;
                    console.error(`HTTP ${status}:`, data);
                } else if (error.request) {
                    // 请求已发送但没有收到响应
                    console.error('网络请求超时或无响应');
                } else {
                    // 其他错误
                    console.error('请求配置错误:', error.message);
                }
                return Promise.reject(error);
            }
        );

        return instance;
    }
};

// 将工具函数暴露到全局
window.TimeUtils = TimeUtils;
window.ValidationUtils = ValidationUtils;
window.FileUtils = FileUtils;
window.FormatUtils = FormatUtils;
window.CameraUtils = CameraUtils;
window.StorageUtils = StorageUtils;
window.ApiUtils = ApiUtils;
