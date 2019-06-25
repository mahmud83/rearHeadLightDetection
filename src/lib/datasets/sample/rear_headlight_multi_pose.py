# -*- coding:UTF-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import numpy as np
import torch
import json
import cv2
import os
import sys

print(os.getcwd())
sys.path.insert(0, os.getcwd())
from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.image import draw_dense_reg
from datasets.dataset.rear_headlight_hp import RearHeadLightHP
from opts import opts
import math


class RearHeadLightMultiPoseDataset(data.Dataset):
    # coco "bbox": [217.62,240.54,38.99,57.75], #[x,y,w,h]
    def _coco_box_to_bbox(self, box):
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                        dtype=np.float32)
        return bbox

    def _get_border(self, border, size):  # 128,w
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i

    def __getitem__(self, index):
        img_id = self.images[index]
        # file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
        # img_path = os.path.join(self.img_dir, file_name)
        # ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        # anns = self.coco.loadAnns(ids=ann_ids)
        annos = self.anno[index]
        img_name = annos['filename'].split('//')[-1]
        img_path = os.path.join(self.img_dir, img_name)
        num_objs = min(len(annos['annotations']), self.max_objs)
        img = cv2.imread(img_path)

        height, width = img.shape[0], img.shape[1]
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)  # center w/2,h/2
        s = max(img.shape[0], img.shape[1]) * 1.0
        rot = 0
        flipped = False
        if self.split == 'train':
            if not self.opt.not_rand_crop:
                s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
                w_border = self._get_border(128, img.shape[1])
                h_border = self._get_border(128, img.shape[0])
                c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
                c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)
            else:
                sf = self.opt.scale
                cf = self.opt.shift
                c[0] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
                c[1] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
                s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
            if np.random.random() < self.opt.aug_rot:
                rf = self.opt.rotate
                rot = np.clip(np.random.randn() * rf, -rf * 2, rf * 2)

            if np.random.random() < self.opt.flip:
                flipped = True
                img = img[:, ::-1, :]
                c[0] = width - c[0] - 1

        trans_input = get_affine_transform(
            c, s, rot, [self.opt.input_w, self.opt.input_h])
        inp = cv2.warpAffine(img, trans_input,
                             (self.opt.input_w, self.opt.input_h),
                             flags=cv2.INTER_LINEAR)
        inp = (inp.astype(np.float32) / 255.)
        if self.split == 'train' and not self.opt.no_color_aug:
            color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
        inp = (inp - self.mean) / self.std
        inp = inp.transpose(2, 0, 1)  # H,W,C -> C,H,W

        # output_res = self.opt.output_res
        num_joints = self.num_joints
        trans_output_rot = get_affine_transform(c, s, rot, [self.opt.output_w, self.opt.output_h])
        trans_output = get_affine_transform(c, s, 0, [self.opt.output_w, self.opt.output_h])

        hm = np.zeros((self.num_classes, self.opt.output_h, self.opt.output_w), dtype=np.float32)  # 1*128*128
        hm_hp = np.zeros((num_joints, self.opt.output_h, self.opt.output_w), dtype=np.float32)  # 17*128*128
        dense_kps = np.zeros((num_joints, 2, self.opt.output_h, self.opt.output_w),
                             dtype=np.float32)
        dense_kps_mask = np.zeros((num_joints, self.opt.output_h, self.opt.output_w),
                                  dtype=np.float32)
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        kps = np.zeros((self.max_objs, num_joints * 2), dtype=np.float32)  # 36,17*2 center offset to hp
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)  # 36*2 center error
        ind = np.zeros((self.max_objs), dtype=np.int64)  # 36
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)  # 36  mask center error
        kps_mask = np.zeros((self.max_objs, self.num_joints * 2), dtype=np.uint8)  # 36,17*2
        hp_offset = np.zeros((self.max_objs * num_joints, 2), dtype=np.float32)  # 36*17,2
        hp_ind = np.zeros((self.max_objs * num_joints), dtype=np.int64)  # 36*17
        hp_mask = np.zeros((self.max_objs * num_joints), dtype=np.int64)  # 36*17

        draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
            draw_umich_gaussian
        gt_det = []
        for k in range(num_objs):
            anno = annos['annotations'][k]
            tempbbox = (anno['x'], anno['y'], anno['width'], anno['height'])
            bbox = self._coco_box_to_bbox(tempbbox)
            cls_id = 0
            # change to coco kpt format
            # v=0 表示这个关键点没有标注（这种情况下x=y=v=0）
            # v=1 表示这个关键点标注了但是不可见(被遮挡了）
            # v=2 表示这个关键点标注了同时也可见
            pts = []
            if anno['headlightVisible'] == -1:
                pts.append(0)
                pts.append(0)
                pts.append(0)
            elif anno['headlightVisible'] == 0:
                pts.append(anno['headlight(h)']['x'])
                pts.append(anno['headlight(h)']['y'])
                pts.append(1)
            elif anno['headlightVisible'] == 1:
                pts.append(anno['headlight(h)']['x'])
                pts.append(anno['headlight(h)']['y'])
                pts.append(2)
            else:
                continue
            pts = np.array(pts, np.float32).reshape(num_joints, 3)
            if flipped:
                bbox[[0, 2]] = width - bbox[[2, 0]] - 1
                pts[:, 0] = width - pts[:, 0] - 1
                for e in self.flip_idx:
                    pts[e[0]], pts[e[1]] = pts[e[1]].copy(), pts[e[0]].copy()
            bbox[:2] = affine_transform(bbox[:2], trans_output)
            bbox[2:] = affine_transform(bbox[2:], trans_output)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, self.opt.output_w - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, self.opt.output_h - 1)
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if (h > 0 and w > 0) or (rot != 0):
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = self.opt.hm_gauss if self.opt.mse_loss else max(0, int(radius))
                ct = np.array(
                    [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)  # w,h
                ct_int = ct.astype(np.int32)
                wh[k] = 1. * w, 1. * h
                ind[k] = ct_int[1] * self.opt.output_w + ct_int[0]  # flatten feature map index
                reg[k] = ct - ct_int
                reg_mask[k] = 1

                # num_kpts = pts[:, 2].sum()
                # if num_kpts == 0:  # no key points,no reg center offset error,no reg w h loss
                #    hm[cls_id, ct_int[1], ct_int[0]] = 0.9999
                #    reg_mask[k] = 0

                hp_radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                hp_radius = self.opt.hm_gauss \
                    if self.opt.mse_loss else max(0, int(hp_radius))
                for j in range(num_joints):
                    if pts[j, 2] > 0:  # valid key point gt
                        pts[j, :2] = affine_transform(pts[j, :2], trans_output_rot)
                        if pts[j, 0] >= 0 and pts[j, 0] < self.opt.output_w and \
                                pts[j, 1] >= 0 and pts[j, 1] < self.opt.output_h:
                            kps[k, j * 2: j * 2 + 2] = pts[j, :2] - ct_int
                            kps_mask[k, j * 2: j * 2 + 2] = 1
                            pt_int = pts[j, :2].astype(np.int32)
                            hp_offset[k * num_joints + j] = pts[j, :2] - pt_int
                            hp_ind[k * num_joints + j] = pt_int[1] * self.opt.output_w + pt_int[0]
                            hp_mask[k * num_joints + j] = 1
                            if self.opt.dense_hp:
                                # must be before draw center hm gaussian
                                draw_dense_reg(dense_kps[j], hm[cls_id], ct_int,
                                               pts[j, :2] - ct_int, radius, is_offset=True)
                                draw_gaussian(dense_kps_mask[j], ct_int, radius)
                            draw_gaussian(hm_hp[j], pt_int, hp_radius)
                draw_gaussian(hm[cls_id], ct_int, radius)
                gt_det.append([ct[0] - w / 2, ct[1] - h / 2,
                               ct[0] + w / 2, ct[1] + h / 2, 1] +
                              pts[:, :2].reshape(num_joints * 2).tolist() + [cls_id])
        if rot != 0:  # no rotation, == 0
            hm = hm * 0 + 0.9999
            reg_mask *= 0
            kps_mask *= 0
        ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh,
               'hps': kps, 'hps_mask': kps_mask}
        if self.opt.dense_hp:
            dense_kps = dense_kps.reshape(num_joints * 2, self.opt.output_h, self.opt.output_w)
            dense_kps_mask = dense_kps_mask.reshape(
                num_joints, 1, self.opt.output_h, self.opt.output_w)
            dense_kps_mask = np.concatenate([dense_kps_mask, dense_kps_mask], axis=1)
            dense_kps_mask = dense_kps_mask.reshape(
                num_joints * 2, self.opt.output_h, self.opt.output_w)
            ret.update({'dense_hps': dense_kps, 'dense_hps_mask': dense_kps_mask})
            del ret['hps'], ret['hps_mask']
        if self.opt.reg_offset:
            ret.update({'reg': reg})
        if self.opt.hm_hp:
            ret.update({'hm_hp': hm_hp})
        if self.opt.reg_hp_offset:
            ret.update({'hp_offset': hp_offset, 'hp_ind': hp_ind, 'hp_mask': hp_mask})

        # if True:
        if self.opt.debug > 0 or not self.split == 'train':
            gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
                np.zeros((1, 40), dtype=np.float32)
            meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': img_id}
            ret['meta'] = meta
        return ret


class TestDataSet(RearHeadLightHP, RearHeadLightMultiPoseDataset):
    pass


if __name__ == '__main__':
    from torch.utils.data import Dataset, DataLoader
    import json

    opt = opts()
    opt = opt.init()
    dataSet = TestDataSet(opt, 'train')
    dataLoader = DataLoader(dataset=dataSet, batch_size=1, shuffle=True)

    d = {}
    for i, ret in enumerate(dataLoader):
        print(i)
        d['hm'] = ret['hm'].data.numpy().tolist()
        d['hm_hp'] = ret['hm_hp'].data.numpy().tolist()
        d['wh'] = ret['wh'].data.numpy().tolist()
        d['img_name'] = ret['meta']['img_name']
        d['c'] = ret['meta']['c'].data.numpy().tolist()
        d['s'] = ret['meta']['s'].data.numpy().tolist()
        d['gt'] = ret['meta']['gt_det'].data.numpy().tolist()
        d = json.dumps(d)
        break
    with open("debug.json", "w") as f:
        f.write(d)
