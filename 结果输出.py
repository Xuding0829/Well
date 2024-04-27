import os
import re
import torch

# 加载YOLOv5模型
model = torch.hub.load('D:/code/python/od/yolov5', 'custom', 'runs/train/exp/weights/best.pt', source='local')

# 输入图像文件夹路径
img_folder = 'D:/code/python/od/dataset/test/images'

# 输出txt文件路径
output_txt = 'predictions.txt'


# 获取图像文件列表并按照名称中的数字部分进行排序
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)


img_files = sorted_alphanumeric(
    [os.path.join(img_folder, f) for f in os.listdir(img_folder) if f.endswith(('.jpg', '.png', '.jpeg'))])

# 打开txt文件以写入预测结果
with open(output_txt, 'w') as f:
    # 遍历每张图片进行目标检测
    for img_path in img_files:
        print("正在处理图像：", img_path)
        # 进行目标检测
        results = model(img_path)

        # 打印检测到的信息
        print("检测到的信息：")
        for det in results.pred[0]:
            class_id = int(det[5])
            confidence = det[4]
            # 将坐标四舍五入为整数
            xmin, ymin, xmax, ymax = map(lambda x: int(round(x.item())), det[:4])
            print(f"类别: {class_id}, 置信度: {confidence:.2f}, 坐标: ({xmin}, {ymin}, {xmax}, {ymax})")
            # 将结果写入txt文件，包含文件后缀
            filename = os.path.splitext(os.path.basename(img_path))[0] + os.path.splitext(img_path)[1]
            line = f"{filename} {class_id} {confidence:.2f} {xmin} {ymin} {xmax} {ymax}\n"
            f.write(line)

        # # 写入换行符以分隔不同图像的结果
        # f.write('\n')

print("预测结果已保存到", output_txt)
