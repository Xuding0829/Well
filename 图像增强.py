# 数据增强选项
self.aug = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),  # 随机亮度对比度
    # A.RandomBrightness(limit=0.3, p=0.5),
    A.GaussianBlur(p=0.3),  # 高斯模糊
    A.GaussNoise(var_limit=(400, 450), mean=0, p=0.7),  # 高斯噪声
    A.CLAHE(clip_limit=2.0, tile_grid_size=(4, 4), p=0.3),  # 直方图均衡
    A.Equalize(p=0.3),  # 均衡图像直方图
    A.Rotate(limit=90, interpolation=0, border_mode=0, p=0.7),  # 旋转
    A.RandomRotate90(p=0.8),
    # A.CoarseDropout(p=0.5),  # 随机生成矩阵黑框
    A.OneOf([
        A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.3),  # RGB图像的每个通道随机移动值
        A.ChannelShuffle(p=0.3),  # 随机排列通道
        A.ColorJitter(p=0.3),  # 随机改变图像的亮度、对比度、饱和度、色调
        A.ChannelDropout(p=0.3),  # 随机丢弃通道
    ], p=0.3),
    A.Downscale(p=0.2),  # 随机缩小和放大来降低图像质量
    A.Emboss(p=0.3),  # 压印输入图像并将结果与原始图像叠加
],
# voc: [xmin, ymin, xmax, ymax]  # 经过归一化
# min_area: 表示bbox占据的像素总个数, 当数据增强后, 若bbox小于这个值则从返回的bbox列表删除该bbox.
# min_visibility: 值域为[0,1], 如果增强后的bbox面积和增强前的bbox面积比值小于该值, 则删除该bbox
A.BboxParams(format='pascal_voc', min_area=0., min_visibility=0., label_fields=['category_id'])
)

print('--------------*--------------')
print("labels: ", self.labels)
if self.start_aug_id is None:
    self.start_aug_id = len(os.listdir(self.pre_xml_path)) + 1
    print("the start_aug_id is not set, default: len(images)", self.start_aug_id)
print('--------------*--------------')
