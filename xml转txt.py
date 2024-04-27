import xml.etree.ElementTree as ET
import os


def convert(size, box):
    width, height = size
    x_center = (box[0] + box[1]) / 2.0
    y_center = (box[2] + box[3]) / 2.0

    # Calculate width and height relative to image size
    w = (box[1] - box[0]) / width
    h = (box[3] - box[2]) / height

    # Normalize x_center, y_center, w, h to range [0, 1]
    x_center = x_center / width
    y_center = y_center / height
    w = w
    h = h

    return x_center, y_center, w, h


def convert_annotation(xml_files_path, save_txt_files_path, classes):
    xml_files = os.listdir(xml_files_path)
    print(xml_files)
    for xml_name in xml_files:
        print(xml_name)
        xml_file = os.path.join(xml_files_path, xml_name)
        out_txt_path = os.path.join(save_txt_files_path, xml_name.split('.')[0] + '.txt')
        with open(out_txt_path, 'w') as out_txt_f:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            size = root.find('size')
            if size is None:
                print(f"Error: 'size' not found in {xml_name}")
                continue
            w = int(size.find('width').text)
            h = int(size.find('height').text)

            for obj in root.iter('object'):
                difficult = obj.find('difficult').text
                cls = obj.find('name').text
                if cls not in classes or int(difficult) == 1:
                    continue
                cls_id = classes.index(cls)
                xmlbox = obj.find('bndbox')
                if xmlbox is None:
                    print(f"Error: 'bndbox' not found in {xml_name}")
                    continue
                b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
                     float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
                bb = convert((w, h), b)
                out_txt_f.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


if __name__ == "__main__":
    classes1 = ['good', 'broke', 'lose', 'uncovered', 'circle']
    xml_files1 = r'D:\Code\Python\od\test_xmls'
    save_txt_files1 = r'D:\Code\Python\od\test_txt'
    convert_annotation(xml_files1, save_txt_files1, classes1)
    with open(os.path.join(save_txt_files1, 'classes.txt'), 'w') as file:
        for class_name in classes1:
            file.write(class_name + '\n')