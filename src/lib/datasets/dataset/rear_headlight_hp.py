from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pycocotools.cocoeval import COCOeval
import pycocotools.coco as coco
import cv2
import numpy as np
import json
import os

import torch.utils.data as data
import sys

sys.path.insert(0, '/home/mingxuzhu/Program_Design/CenterNet/src/lib/')
from opts import opts


class RearHeadLightHP(data.Dataset):
    num_classes = 1
    num_joints = 1
    default_resolution = [448, 832]
    mean = np.array([0.40789654, 0.44719302, 0.47026115],  # should accord to rear dataSet cal
                    dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.28863828, 0.27408164, 0.27809835],  # should accord to rear dataSet cal
                   dtype=np.float32).reshape(1, 1, 3)
    flip_idx = []

    def __init__(self, opt, split):
        super(RearHeadLightHP, self).__init__()
        # self.edges = [[0, 1], [0, 2], [1, 3], [2, 4],
        #              [4, 6], [3, 5], [5, 6],
        #              [5, 7], [7, 9], [6, 8], [8, 10],
        #              [6, 12], [5, 11], [11, 12],
        #              [12, 14], [14, 16], [11, 13], [13, 15]]
        # self.acc_idxs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

        self.data_dir = os.path.join(opt.data_dir, 'rear_headlight')
        self.img_dir = os.path.join(self.data_dir, 'images', '{}'.format(split))
        if split == 'test':
            self.annot_path = os.path.join(
                self.data_dir, 'annotations',
                'test.json').format(split)
        else:
            self.annot_path = os.path.join(
                self.data_dir, 'annotations',
                '{}.json').format(split)  # train.json
        self.max_objs = 32
        self._data_rng = np.random.RandomState(123)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                                 dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)
        self.split = split
        self.opt = opt

        print('==> initializing rear headlight {} data.'.format(split))
        self.anno = json.load(open(self.annot_path))
        self._convert_eval_anno_format()

        self.coco = coco.COCO(os.path.join(self.data_dir, 'annotations', 'coco_annotations.json'))
        image_ids = self.coco.getImgIds()
        self.num_samples = len(self.anno)
        if split == 'train':
            self.images = []
            for img_id in image_ids:
                idxs = self.coco.getAnnIds(imgIds=[img_id])
                if len(idxs) > 0:
                    self.images.append(img_id)
        else:
            self.images = image_ids

        print('Loaded {} {} samples'.format(split, self.num_samples))

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def _convert_eval_anno_format(self):

        res = self.anno
        new_anno = {}
        categories = [{'supercategory': 'vehicle(v)', 'id': 1, 'name': 'vehicle(v)',
                       'keypoints': ['right_headlight'], 'skeleton': []}]
        new_anno = {'categories': categories}
        num_images = len(res)
        images = []
        for i in range(num_images):
            img_info = {}
            file_name = res[i]["filename"].split("//")[-1]
            img = cv2.imread(os.path.join(self.img_dir, file_name))
            height, width = img.shape[0], img.shape[1]
            img_info['file_name'] = file_name
            img_info['height'] = height
            img_info['width'] = width
            img_info['id'] = i  # 0..19
            images.append(img_info)
        new_anno['images'] = images
        annotations = []
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
        anno_dump = json.dumps(new_anno)
        with open(os.path.join(self.data_dir, "annotations", "coco_annotations.json"), "w") as fp:
            fp.write(anno_dump)
        return new_anno

    def convert_eval_format(self, all_bboxes):
        # import pdb; pdb.set_trace()
        detections = []
        for image_id in all_bboxes:
            for cls_ind in all_bboxes[image_id]:
                category_id = 1
                for dets in all_bboxes[image_id][cls_ind]:
                    bbox = dets[:4]
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]
                    score = dets[4]
                    bbox_out = list(map(self._to_float, bbox))
                    keypoints = np.concatenate([
                        np.array(dets[5:7], dtype=np.float32).reshape(-1, 2),
                        np.ones((1, 1), dtype=np.float32)], axis=1).reshape(3).tolist()
                    keypoints = list(map(self._to_float, keypoints))

                    detection = {
                        "image_id": int(image_id),
                        "category_id": int(category_id),
                        "bbox": bbox_out,
                        "score": float("{:.2f}".format(score)),
                        "keypoints": keypoints
                    }
                    detections.append(detection)
        return detections

    def __len__(self):
        return self.num_samples

    def save_results(self, results, save_dir):
        json.dump(self.convert_eval_format(results),
                  open('{}/results.json'.format(save_dir), 'w'))

    def run_eval(self, results, save_dir):
        # result_json = os.path.join(opt.save_dir, "results.json")
        # detections  = convert_eval_format(all_boxes)
        # json.dump(detections, open(result_json, "w"))
        self.save_results(results, save_dir)
        coco_dets = self.coco.loadRes('{}/results.json'.format(save_dir))
        coco_eval = COCOeval(self.coco, coco_dets, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        coco_eval = COCOeval(self.coco, coco_dets, "keypoints")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()


if __name__ == '__main__':
    opt = opts()
    opt = opt.init()
    dataSet = RearHeadLightHP(opt, 'train')
