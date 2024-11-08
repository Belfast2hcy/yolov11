import argparse
import os
import xml.etree.ElementTree as ET
from ultralytics import YOLO # 根据你的 YOLO 版本来引入相应的库


def process_yolo_model(input_dir, model_file):
    model = YOLO(model_file)
    results = {}

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.jpg', '.png')):  # 确保不区分大小写
            file_path = os.path.join(input_dir, filename)
            try:
                detections = model.predict(file_path)
                results[filename] = detections
            except Exception as e:
                print(f"处理 {filename} 时出错: {e}")

    return results


def generate_xml(results, output_dir):
    for key, value in results.items():
        output_file = os.path.join(output_dir, f"{key}.xml")

        for item in value:
            box = item.boxes  # 获取 boxes 属性
        root = ET.Element("annotation")
        # 添加边界框和其他信息
        for row in range(len(box.xyxy)):
            detection = ET.SubElement(root, "object")
            bndbox = ET.SubElement(detection, "bndbox")
            ET.SubElement(bndbox, "xmin").text = str(int(box.xyxy[row][0]))
            ET.SubElement(bndbox, "ymin").text = str(int(box.xyxy[row][1]))
            ET.SubElement(bndbox, "xmax").text = str(int(box.xyxy[row][2]))
            ET.SubElement(bndbox, "ymax").text = str(int(box.xyxy[row][3]))
            ET.SubElement(bndbox, "confidence").text = str(int(box.conf[row]))
            ET.SubElement(bndbox, "class").text = str(int(box.cls[row]))

        # 创建 XML 树并写入文件
        tree = ET.ElementTree(root)
        tree.write(output_file)

def main():
    parser = argparse.ArgumentParser(description='Process YOLO model.')
    parser.add_argument('-dir', required=True, help='Input directory containing images')
    parser.add_argument('-model', required=True, help='YOLO model file path')
    parser.add_argument('-out', required=True, help='Output directory for XML files')

    args = parser.parse_args()

    results = process_yolo_model(args.dir, args.model)
    generate_xml(results, args.out)

    print(f"Results saved to: {args.out}")


if __name__ == "__main__":
    main()
