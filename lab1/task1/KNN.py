import cv2
import os
import numpy as np
import os.path as osp
from skimage import io
import random
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import joblib


def HOG_descriptor(image):
    if (image.max() - image.min()) != 0:
        image = (image - image.min()) / (image.max() - image.min())
        image *= 255
        image = image.astype(np.uint8)
    HOG = cv2.HOGDescriptor((64, 128), (16, 16), (8, 8), (8, 8), 9)
    HOG_feature = HOG.compute(image)
    return HOG_feature


# 导入图像
poslist = os.listdir('D:/AI learning/contentSecurity/lab1/INRIA/train/pos')
neglist = os.listdir('D:/AI learning/contentSecurity/lab1/INRIA/train/neg')
testlist = os.listdir('D:/AI learning/contentSecurity/lab1/INRIA/test/pos')
testnlist = os.listdir('D:/AI learning/contentSecurity/lab1/INRIA/test/neg')

# 获得正样本和负样本的HOG特征，并标记
HOG_list = []
label_list = []
print("正样本图像有" + str(len(poslist)))
print("负样本原始图像有" + str(len(neglist)) + "，每个原始图像提供十个负样本")
for i in range(len(poslist)):
    posimg = io.imread(osp.join('D:/AI learning/contentSecurity/lab1/INRIA/train/pos', poslist[i]))
    posimg = cv2.cvtColor(posimg, cv2.COLOR_RGBA2BGR)  # 将一个RGBA格式的图像转换为BGR格式的图像
    # 所用图像已经经过标准化
    posimg = cv2.resize(posimg, (64, 128),
                        interpolation=cv2.INTER_NEAREST)  # 将图像posimg调整为指定大小，即将图像的宽度调整为64像素，高度调整为128像素。调整过程中使用最近邻插值法进行插值处理。
    pos_HOG = HOG_descriptor(posimg)
    HOG_list.append(pos_HOG)
    label_list.append(1)
for i in range(len(neglist)):
    negimg = io.imread(osp.join('D:/AI learning/contentSecurity/lab1/INRIA/train/neg', neglist[i]))
    negimg = cv2.cvtColor(negimg, cv2.COLOR_RGBA2BGR)

    # 在每张negimg图像中截取10张标准大小的图片作为负样本
    for j in range(10):
        y = int(random.random() * (negimg.shape[0] - 128))
        x = int(random.random() * (negimg.shape[1] - 64))
        negimgs = negimg[y:y + 128, x:x + 64]
        negimgs = cv2.resize(negimgs, (64, 128), interpolation=cv2.INTER_NEAREST)
        neg_HOG = HOG_descriptor(negimgs)
        HOG_list.append(neg_HOG)
        label_list.append(0)
print(type(HOG_list[10]))
print(type(HOG_list[-10]))
HOG_list = np.float32(HOG_list)
label_list = np.int32(label_list).reshape(len(label_list), 1)

# 训练KNN，并保存模型
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(HOG_list.squeeze(), label_list.squeeze())
joblib.dump(knn, "trained_knn.m")

# 提取训练集样本和标签
test_HOG = []
test_label = []
for i in range(len(testlist)):
    testimg = io.imread(osp.join('D:/AI learning/contentSecurity/lab1/INRIA/test/pos', testlist[i]))
    testimg = cv2.cvtColor(testimg, cv2.COLOR_RGBA2BGR)
    testimg = cv2.resize(testimg, (64, 128), interpolation=cv2.INTER_NEAREST)
    testhog = HOG_descriptor(testimg)
    test_HOG.append(testhog)
    test_label.append(1)

for i in range(len(testnlist)):
    testnegimg = io.imread(osp.join('D:/AI learning/contentSecurity/lab1/INRIA/test/neg', testnlist[i]))
    testnegimg = cv2.cvtColor(testnegimg, cv2.COLOR_RGBA2BGR)

    # 在每张negimg图像中截取10张标准大小的图片作为负样本
    for j in range(10):
        y = int(random.random() * (testnegimg.shape[0] - 128))
        x = int(random.random() * (testnegimg.shape[1] - 64))
        testnegimgs = testnegimg[y:y + 128, x:x + 64]
        testnegimgs = cv2.resize(testnegimgs, (64, 128), interpolation=cv2.INTER_NEAREST)
        testneghog = HOG_descriptor(testnegimgs)
        test_HOG.append(testneghog)
        test_label.append(0)
test_HOG = np.float32(test_HOG)
test_label = np.int32(test_label).reshape(len(test_label), 1)

# 加载训练好的KNN模型
knn = joblib.load("trained_knn.m")

# 对测试集进行预测并计算评估指标
from sklearn.metrics import confusion_matrix, average_precision_score, roc_curve, roc_auc_score

# 获取概率预测值
prob = knn.predict_proba(test_HOG.squeeze())[:, 1]

# 获取类别预测值
y_pred = knn.predict(test_HOG.squeeze())

# 计算混淆矩阵
cm = confusion_matrix(test_label.squeeze(), y_pred)
print("混淆矩阵:")
print(cm)
tn, fp, fn, tp = cm.ravel()
print(f"真阴性(TN): {tn}, 假阳性(FP): {fp}, 假阴性(FN): {fn}, 真阳性(TP): {tp}")

# 计算平均精确率(AP)
AP = average_precision_score(test_label.squeeze(), prob)
print(f"平均精确率(AP): {AP:.4f}")

# 计算ROC曲线并绘制
fpr, tpr, thresholds = roc_curve(test_label.squeeze(), prob, pos_label=1)
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, 'b-', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--')  # 随机猜测的基准线
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.grid(True)
plt.savefig('D:/AI learning/contentSecurity/lab1/task1/ROC.png', dpi=300)

# 计算AUC
AUC = roc_auc_score(test_label.squeeze(), prob)
print(f"ROC曲线下面积(AUC): {AUC:.4f}")

# 计算其他常用指标
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
miss_rate = fn / (tp + fn) if (tp + fn) > 0 else 0

print(f"准确率(Accuracy): {accuracy:.4f}")
print(f"精确率(Precision): {precision:.4f}")
print(f"召回率(Recall): {recall:.4f}")
print(f"特异度(Specificity): {specificity:.4f}")
print(f"漏检率(Miss Rate): {miss_rate:.4f}")