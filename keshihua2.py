from lxml import etree
import cv2


# 读取 xml 文件信息，并返回字典形式
def parse_xml_to_dict(xml):
    if len(xml) == 0:  # 遍历到底层，直接返回 tag对应的信息
        return {xml.tag: xml.text}

    result = {}
    for child in xml:
        child_result = parse_xml_to_dict(child)  # 递归遍历标签信息
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:  # 因为object可能有多个，所以需要放入列表里
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}


# xml 标注文件的可视化
def xmlShow(img, xml, save=True):
    image = cv2.imread(img)
    if image is None:
        print("Error: Could not read the image.")
        return

    try:
        with open(xml, encoding='gb18030', errors='ignore') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
    except Exception as e:
        print(f"Error reading XML: {e}")
        return

    try:
        data = parse_xml_to_dict(xml)['last/train_0351.xml']
    except KeyError:
        print("Error: Could not find expected data in XML.")
        return
    with open(xml, encoding='gb18030', errors='ignore') as fid:  # 防止出现非法字符报错
        xml_str = fid.read()
    xml = etree.fromstring(xml_str)
    data = parse_xml_to_dict(xml)['last/train_0351.xml']  # 读取 xml文件信息

    ob = []  # 存放目标信息
    for i in data['object']:  # 提取检测框
        name = str(i['name'])  # 检测的目标类别

        bbox = i['bndbox']
        xmin = int(bbox['xmin'])
        ymin = int(bbox['ymin'])
        xmax = int(bbox['xmax'])
        ymax = int(bbox['ymax'])

        tmp = [name, xmin, ymin, xmax, ymax]  # 单个检测框
        ob.append(tmp)

    # 绘制检测框
    for name, x1, y1, x2, y2 in ob:
        cv2.rectangle(image, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)  # 绘制矩形框
        cv2.putText(image, name, (x1, y1 - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5, thickness=1, color=(0, 0, 255))

    # 保存图像
    if save:
        cv2.imwrite('result.png', image)

    # 展示图像
    cv2.imshow('test', image)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    img_path = 'test/train_0351.jpg'  # 传入图片

    labels_path = img_path.replace('images', 'labels')  # 自动获取对应的 xml 标注文件
    labels_path = labels_path.replace('.jpeg', '.xml')

    xmlShow(img=img_path, xml=labels_path, save=True)