import os
import cv2

# 设置图像文件夹路径
image_folder = r'G:\python\cnn_catanddog\data\train'

# 设置目标大小
target_size = (224, 224)  # 替换为你想要的目标大小

# 获取文件夹中的所有图像文件
img_path_list=os.listdir(image_folder)

# 遍历每张图像
for image_file in img_path_list:
    # 构建图像文件的完整路径
    image_path = os.path.join(image_folder, image_file)

    # 读取图像
    image = cv2.imread(image_path)

    # 调整图像大小
    resized_image = cv2.resize(image, target_size)

    # 覆盖原有图像
    cv2.imwrite(image_path, resized_image)

print("图像大小调整完成！")
