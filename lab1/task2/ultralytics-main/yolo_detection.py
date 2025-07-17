from ultralytics import YOLO

model = YOLO("yolov8n.pt")
results = model.train(data="coco128.yaml", epochs=5)

# 在验证集上评估模型性能
metrics = model.val()

# 使用训练好的模型进行预测
results = model("D:/AI learning/contentSecurity/lab1/task2/image.jpg")
results[0].plot()