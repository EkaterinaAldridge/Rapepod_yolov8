from ultralytics import YOLO
import numpy as np
import cv2
import random
import os
from tqdm import tqdm

# mask颜色深度
MASK_DEPTH = 0.2
# 坐标偏置
X_OFFSET = 0
Y_OFFSET = 23
# 文字颜色
WORD_COLOR = (255, 204, 0)  # 天蓝色


# 设置随机颜色
def randColor():
    return (random.uniform(1, 255), random.uniform(1, 255), random.uniform(1, 255))


def get_plots(img, bboxes, plots):
    for bbox, plot in (zip(bboxes, plots)):
        # 获取mask点坐标
        point = (plot * np.array([ori_shape[1], ori_shape[0]])).astype(np.int32)
        # 掩码
        zeros = np.zeros((img.shape), dtype=np.uint8)
        color = randColor()
        mask = cv2.fillPoly(zeros, [point], color=color)
        img = MASK_DEPTH * mask + img
        # 画框
        bbox = bbox.astype(np.int32)
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        # 计算长宽比
        area = cv2.contourArea(point, False)
        perimeter = cv2.arcLength(point, True)
        length = perimeter / 2
        width = area / length
        ratio = round(length / width, 2)
        # 获取框中心点坐标
        o_x, o_y = int((bbox[2] + bbox[0]) / 2), int((bbox[3] + bbox[1]) / 2)
        # 将长宽比写入图中
        text = f'ratio:{ratio}'
        cv2.putText(img, text, (bbox[0] + X_OFFSET, bbox[1] + Y_OFFSET), cv2.FONT_HERSHEY_PLAIN, 2.0, WORD_COLOR, 2)    # (o_x - X_OFFSET, o_y + Y_OFFSET)

    return img


if __name__ == '__main__':
    # 模型路径
    model_path = 'rape_model/best.pt'
    model = YOLO(model_path)
    # 图片路径
    # imgs_path = "datasets/rapedata/test"
    imgs_path = 'datasets/rapedata/images'
    # 结果保存路径
    save_path = "outputs"
    os.makedirs(save_path, exist_ok=True)

    imgs_list = tqdm(os.listdir(imgs_path))
    for img_name in imgs_list:
        try:
            img_path = os.path.join(imgs_path, img_name)
            # 预测模型（保存txt文件）
            result = model(img_path, save_txt=True)[0]
            img = cv2.imread(img_path)
            bboxes = result.boxes.xyxy.cpu().numpy()
            plots = result.masks.segments
            ori_shape = result.orig_shape

            result_img = get_plots(img, bboxes, plots)
            save_file_path = os.path.join(save_path, img_name)
            cv2.imwrite(save_file_path, result_img)
        except:
            print(f"图片处理失败：{img_name}")
            continue

