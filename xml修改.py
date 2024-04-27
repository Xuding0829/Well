import os
import xml.etree.ElementTree as ET
from PIL import Image

def fix_xml_size(xml_file, image_folder):
    # Parse XML
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Find size element
    size_elem = root.find('size')
    if size_elem is None:
        print(f"Error: 'size' element not found in {xml_file}")
        return False

    # Get image filename
    image_filename = os.path.splitext(os.path.basename(xml_file))[0] + '.jpg'
    image_path = os.path.join(image_folder, image_filename)

    # Check if image file exists
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} not found")
        return False

    # Get image size
    image = Image.open(image_path)
    width, height = image.size

    # Update size element
    size_elem.find('width').text = str(width)
    size_elem.find('height').text = str(height)

    # Save updated XML
    tree.write(xml_file)

    print(f"Fixed size for {xml_file}: {width} x {height}")
    return True


def fix_xmls_with_zero_size(xml_folder, image_folder):
    for xml_file in os.listdir(xml_folder):
        if xml_file.endswith('.xml'):
            xml_path = os.path.join(xml_folder, xml_file)
            try:
                if not fix_xml_size(xml_path, image_folder):
                    print(f"Failed to fix {xml_path}")
            except Exception as e:
                print(f"Error processing {xml_path}: {e}")


if __name__ == "__main__":
    xml_folder = r'D:\Download\2024中国大学生服务外包创新创业大赛data\train_xmls'
    image_folder = r'D:\Download\2024中国大学生服务外包创新创业大赛data\train_images'

    fix_xmls_with_zero_size(xml_folder, image_folder)