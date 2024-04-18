import glob
import os
import json


# 将json分割转换成yolo格式
def json2yolo(json_file, txt_file, classes_id):
    with open(json_file, 'r') as f:
        anno = json.load(f)
    w = anno['imageWidth']
    h = anno['imageHeight']
    shapes = anno['shapes']
    with open(txt_file, 'w') as txt:
        for shape in shapes:
            label = shape['label']
            points = shape['points']
            # print(label, classes_id)
            if label not in classes_id:
                continue
            label_id = classes_id.index(label)
            line = [label_id]
            for n, point in enumerate(points):
                x = round(point[0] / w, 6)
                y = round(point[1] / h, 6)
                line.append(x)
                line.append(y)
            txt.write(('%g ' * len(line)).rstrip() % tuple(line) + '\n')


if __name__ == '__main__':
    data_path = "rapedata"
    json_list = glob.glob(os.path.join(data_path, '*.json'))
    print(json_list)
    os.makedirs('rapedata_label', exist_ok=True)
    classes_id = ["1"]
    for json_file in json_list:
        txt_name = os.path.basename(json_file[:-4] + 'txt')
        txt_file = os.path.join('rapedata_label', txt_name)
        json2yolo(json_file, txt_file, classes_id)
    print('done')
