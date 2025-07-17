import idx2numpy
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 加载MNIST测试集图像和标签
x_test = idx2numpy.convert_from_file('t10k-images.idx3-ubyte')
y_test = idx2numpy.convert_from_file('t10k-labels.idx1-ubyte')

# 将图像像素值归一化到0和1之间
x_test = x_test.reshape(-1, 28, 28, 1).astype(np.float32) / 255.0

# 对标签进行独热编码
y_test = tf.keras.utils.to_categorical(y_test, 10)

# 将独热编码的标签转换为TensorFlow张量
y_test_tensor = tf.convert_to_tensor(y_test, dtype=tf.float32)

# 加载预训练的模型
model = tf.keras.models.load_model('lenet_model.h5')


def generate_adversarial_examples(model, x, y_true, epsilon=0.01):
    # 使用tf.GradientTape记录梯度信息
    with tf.GradientTape() as tape:
        tape.watch(x)
        # 获取模型在输入样本上的预测结果
        y_pred = model(x)
        # 计算损失函数
        loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)

    # 计算损失函数对输入样本的梯度
    gradients = tape.gradient(loss, x)
    # 生成对抗样本
    x_adversarial = x + epsilon * tf.sign(gradients)
    x_adversarial = tf.clip_by_value(x_adversarial, 0, 1)

    return x_adversarial


def compute_accuracy(y_true, y_pred):
    y_pred_label = tf.argmax(y_pred, axis=1)
    y_true_label = tf.argmax(y_true, axis=1)
    correct_prediction = tf.equal(y_pred_label, y_true_label)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy.numpy()


# 设置不同的epsilon值
epsilons = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

# 存储准确率
accuracies = []

# 将测试图像转换为TensorFlow张量
x_test_tensor = tf.convert_to_tensor(x_test, dtype=tf.float32)

for epsilon in epsilons:
    # 生成对抗样本
    x_adversarial = generate_adversarial_examples(model, x_test_tensor, y_test_tensor, epsilon)

    # 获取模型在对抗样本上的预测结果
    y_pred_adversarial = model.predict(x_adversarial)

    # 计算准确率
    accuracy = compute_accuracy(y_test_tensor, y_pred_adversarial)
    accuracies.append(accuracy)

    print(f'Epsilon: {epsilon}, Accuracy: {accuracy}')

# 绘制准确率曲线
plt.plot(epsilons, accuracies)
plt.xlabel('Epsilon')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Epsilon')
plt.show()