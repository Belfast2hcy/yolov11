from ultralytics import YOLO
import matplotlib.pyplot as plt
import os


def main():
    try:
        # Load the trained model
        model = YOLO(r"D:\深度学习\ultralytics-main\runs\detect\train5\weights\best.pt")  # 加载训练好的模型

        # Run inference on an image
        img_path = r"D:\深度学习\ultralytics-main\test\train_0361.JPG"  # 替换为您的输入图像路径
        results = model.predict(source=img_path)  # 进行推理

        # Print results
        for result in results:
            boxes = result.boxes  # 获取边界框信息
            for box in boxes:
                print("Predicted box coordinates:", box.xyxy)  # 打印边界框坐标
                print("Confidence score:", box.conf)  # 打印置信度
                print("Class ID:", box.cls)  # 打印类别ID

        # Render the predictions on the image
        annotated_img = results[0].plot()  # 获取带检测框的图像

        # Display the results
        plt.imshow(annotated_img)  # 显示带检测框的图像，保持原图颜色
        plt.axis('off')  # 不显示坐标轴
        plt.show()

        # Save the annotated image
        output_path = "D:/深度学习/ultralytics-main/last/annotated_image.jpg"  # 输出路径，确保包含文件名
        os.makedirs(os.path.dirname(output_path), exist_ok=True)  # 创建输出目录（如果不存在的话）

        # Save the annotated image without changing its colors
        plt.imsave(output_path, annotated_img)  # 使用plt.imsave保存图像
        print(f"Annotated image saved to {output_path}")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
