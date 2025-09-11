import cv2
import os
import glob
import numpy as np

def convert_png_8uc3_to_16uc1(input_folder, output_folder):
    """
    读取一个文件夹下所有的CV_8UC3 PNG文件，将它们转换为CV_16UC1，
    并保存到另一个文件夹。

    参数:
    input_folder (str): 包含原始PNG图像的文件夹路径。
    output_folder (str): 用于保存转换后图像的文件夹路径。
    """
    # 步骤 1: 检查并创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"创建文件夹: {output_folder}")

    # 步骤 2: 获取输入文件夹下所有的PNG文件路径
    # 使用glob模块来匹配所有.png文件
    search_path = os.path.join(input_folder, '*.png')
    image_paths = glob.glob(search_path)

    if not image_paths:
        print(f"在文件夹 {input_folder} 中未找到PNG文件。")
        return

    print(f"找到 {len(image_paths)} 个PNG文件。开始转换...")

    # 步骤 3: 遍历所有图片，进行转换和保存
    for img_path in image_paths:
        try:
            # 读取图像。默认情况下，imread以BGR格式（CV_8UC3）读取彩色图像
            img_8uc3 = cv2.imread(img_path, cv2.IMREAD_COLOR)

            if img_8uc3 is None:
                print(f"警告: 无法读取文件 {img_path}，跳过。")
                continue

            # 步骤 3a: 将BGR图像转换为灰度图像 (从3通道变为1通道)
            img_8uc1 = cv2.cvtColor(img_8uc3, cv2.COLOR_BGR2GRAY)

            # 步骤 3b: 将8位图像转换为16位图像 (CV_8UC1 -> CV_16UC1)
            # 为了将0-255的范围正确映射到0-65535，我们将每个像素值乘以257 (65535/255)
            # 使用np.uint16确保数据类型正确
            img_16uc1 = (img_8uc1.astype(np.uint16)) * 257

            # 步骤 4: 构建输出文件路径并保存图像
            # 获取原始文件名
            file_name = os.path.basename(img_path)
            # 构建完整的保存路径
            save_path = os.path.join(output_folder, file_name)

            # 以PNG格式保存16位图像
            cv2.imwrite(save_path, img_16uc1)

            print(f"成功转换并保存: {img_path} -> {save_path}")

        except Exception as e:
            print(f"处理文件 {img_path} 时发生错误: {e}")

    print("所有转换已完成。")

# --- 使用示例 ---
if __name__ == '__main__':
    # 设置你的输入和输出文件夹路径
    # 请确保将 'path/to/your/input_images' 替换为你的实际路径
    input_directory = 'F:\AI\\test_chejian'
    # 请确保将 'path/to/your/output_images' 替换为你的期望路径
    output_directory = 'F:\AI\\test_chejian\\16UC1'

    # 检查输入路径是否存在
    if not os.path.isdir(input_directory):
        print(f"错误: 输入文件夹 '{input_directory}' 不存在。请检查路径。")
    else:
        convert_png_8uc3_to_16uc1(input_directory, output_directory)