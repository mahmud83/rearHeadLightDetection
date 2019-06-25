import json
import cv2

path = "/home/mingxuzhu/Program_Design/CenterNet/data/rear_headlight/annotations/train.json"

res = json.load(open(path))

img_dir = "/home/mingxuzhu/Program_Design/CenterNet/data/rear_headlight/images/train/"

new_anno = {}
categories = [{'supercategory': 'vehicle(v)', 'id': 1, 'name': 'vehicle(v)',
               'keypoints': ['right_headlight'], 'skeleton': []}]
new_anno = {'categories': categories}

num_images = len(res)
images = []

for i in range(num_images):
    img_info = {}
    file_name = res[i]["filename"].split("//")[-1]
    img = cv2.imread(img_dir + file_name)
    height, width = img.shape[0], img.shape[1]
    img_info['file_name'] = file_name
    img_info['height'] = height
    img_info['width'] = width
    img_info['id'] = i  # 0..19
    images.append(img_info)
new_anno['images'] = images

annotations = []

# {'segmentation': [
#     [125.12, 539.69, 140.94, 522.43, 100.67, 496.54, 84.85, 469.21, 73.35, 450.52, 104.99, 342.65, 168.27, 290.88,
#      179.78, 288, 189.84, 286.56, 191.28, 260.67, 202.79, 240.54, 221.48, 237.66, 248.81, 243.42, 257.44, 256.36,
#      253.12, 262.11, 253.12, 275.06, 299.15, 233.35, 329.35, 207.46, 355.24, 206.02, 363.87, 206.02, 365.3, 210.34,
#      373.93, 221.84, 363.87, 226.16, 363.87, 237.66, 350.92, 237.66, 332.22, 234.79, 314.97, 249.17, 271.82, 313.89,
#      253.12, 326.83, 227.24, 352.72, 214.29, 357.03, 212.85, 372.85, 208.54, 395.87, 228.67, 414.56, 245.93, 421.75,
#      266.07, 424.63, 276.13, 437.57, 266.07, 450.52, 284.76, 464.9, 286.2, 479.28, 291.96, 489.35, 310.65, 512.36,
#      284.76, 549.75, 244.49, 522.43, 215.73, 546.88, 199.91, 558.38, 204.22, 565.57, 189.84, 568.45, 184.09, 575.64,
#      172.58, 578.52, 145.26, 567.01, 117.93, 551.19, 133.75, 532.49]], 'num_keypoints': 10, 'area': 47803.27955,
#     'iscrowd': 0,
#     'keypoints': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 142, 309, 1, 177, 320, 2, 191, 398, 2, 237, 317, 2, 233,
#                   426, 2, 306, 233, 2, 92, 452, 2, 123, 468, 2, 0, 0, 0, 251, 469, 2, 0, 0, 0, 162, 551, 2],
#     'image_id': 425226, 'bbox': [73.35, 206.02, 300.58, 372.5], 'category_id': 1, 'id': 183126}
anno_id = 0
for image_id, anno in enumerate(res):
    for _, obj_info in enumerate(anno['annotations']):
        new_obj_info = {}
        bbox = [obj_info['x'], obj_info['y'], obj_info['width'], obj_info['height']]
        keypoints = []
        new_obj_info['segmentation'] = []
        new_obj_info['num_keypoints'] = 1
        new_obj_info['area'] = obj_info['width'] * obj_info['height']
        new_obj_info['iscrowd'] = 0

        if obj_info['headlightVisible'] == -1:
            keypoints = [0, 0, 0]
        elif obj_info['headlightVisible'] == 0:
            keypoints = [obj_info['headlight(h)']['x'], obj_info['headlight(h)']['y'], 1]
        elif obj_info['headlightVisible'] == 1:
            keypoints = [obj_info['headlight(h)']['x'], obj_info['headlight(h)']['y'], 2]
        else:
            continue
        new_obj_info['keypoints'] = keypoints
        new_obj_info['image_id'] = image_id
        new_obj_info['id'] = anno_id
        new_obj_info['bbox'] = bbox
        new_obj_info['category_id'] = 1
        annotations.append(new_obj_info)
        anno_id += 1
new_anno['annotations'] = annotations

