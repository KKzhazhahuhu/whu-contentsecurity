import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

import librosa
import librosa.display
import warnings

warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical

# 设置随机种子，确保结果可复现
np.random.seed(42)
tf.random.set_seed(42)


def load_metadata(metadata_path):
    """加载数据集元数据"""
    metadata = pd.read_csv(metadata_path)
    print(f"数据集大小: {len(metadata)}")
    print(f"类别数量: {len(metadata.classID.unique())}")
    return metadata


def extract_features(file_path, max_length=44100 * 4):
    """从音频文件中提取MFCC特征"""
    try:
        # 加载音频文件
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast', duration=4)

        # 填充或截断音频到固定长度
        if len(audio) < max_length:
            audio = np.pad(audio, (0, max_length - len(audio)), 'constant')
        else:
            audio = audio[:max_length]

        # 提取梅尔频率倒谱系数(MFCC)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)

        # 规范化MFCC
        mfccs = (mfccs - np.mean(mfccs)) / np.std(mfccs)

        return mfccs
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None


def prepare_dataset(metadata, data_dir, test_fold=10):
    """准备训练和测试数据集"""
    valid_indices = []
    features = []
    labels = []

    for idx, row in metadata.iterrows():
        file_path = os.path.join(data_dir, f'fold{row["fold"]}', row["slice_file_name"])
        if not os.path.exists(file_path):
            print(f"警告: 文件不存在 {file_path}")
            continue

        mfccs = extract_features(file_path)

        if mfccs is not None:
            features.append(mfccs)
            labels.append(row["classID"])
            valid_indices.append(idx)

    features = np.array(features)
    labels = np.array(labels)
    features = features[..., np.newaxis]

    test_indices = []
    train_indices = []

    for i, idx in enumerate(valid_indices):
        if metadata.iloc[idx]['fold'] == test_fold:
            test_indices.append(i)
        else:
            train_indices.append(i)

    X_train = features[train_indices]
    y_train = labels[train_indices]
    X_test = features[test_indices]
    y_test = labels[test_indices]

    print(f"训练集大小: {len(X_train)}")
    print(f"测试集大小: {len(X_test)}")

    # 对标签进行独热编码
    num_classes = len(np.unique(labels))
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    return X_train, y_train, X_test, y_test, num_classes


def build_cnn_model(input_shape, num_classes):
    """构建CNN模型用于音频分类"""
    model = Sequential([
        # 第一个卷积块
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # 第二个卷积块
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # 第三个卷积块
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # 分类器
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    # 编译模型
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()
    return model


def train_and_evaluate(X_train, y_train, X_test, y_test, model, epochs=50, batch_size=32):
    """训练并评估模型"""
    # 设置回调函数
    checkpoint = ModelCheckpoint(
        'best_model.h5',
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=1,
        restore_best_weights=True
    )

    # 训练模型
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_test, y_test),
        callbacks=[checkpoint, early_stopping]
    )

    # 评估模型
    score = model.evaluate(X_test, y_test, verbose=0)
    print(f"测试集准确率: {score[1] * 100:.2f}%")

    # 获取混淆矩阵
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    # 打印分类报告
    class_names = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling',
                   'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music']
    print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))

    # 绘制混淆矩阵
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()

    # 绘制训练历史
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

    return history


# 4. 主函数
def main():
    """主函数，运行整个实验流程"""
    # 设置GPU内存增长
    physical_devices = tf.config.list_physical_devices('GPU')
    for device in physical_devices:
        try:
            tf.config.experimental.set_memory_growth(device, True)
        except:
            pass

    # 设置数据路径
    data_dir = 'UrbanSound8K_reduced/audio'
    metadata_path = 'UrbanSound8K_reduced/UrbanSound8K_reduced.csv'

    # 加载元数据
    metadata = load_metadata(metadata_path)

    # 数据集太小时，减少训练轮次和增大批次大小
    epochs = 5
    batch_size = 16

    # 准备数据集 (使用第10折作为测试集)
    X_train, y_train, X_test, y_test, num_classes = prepare_dataset(metadata, data_dir, test_fold=10)
    print(f"训练集形状: {X_train.shape}, 测试集形状: {X_test.shape}")

    # 构建模型
    input_shape = X_train[0].shape
    model = build_cnn_model(input_shape, num_classes)

    # 训练并评估模型
    history = train_and_evaluate(X_train, y_train, X_test, y_test, model, epochs=epochs, batch_size=batch_size)

    print("实验完成!")


if __name__ == "__main__":
    main()