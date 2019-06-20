from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .sample.ddd import DddDataset
from .sample.exdet import EXDetDataset
from .sample.ctdet import CTDetDataset
from .sample.multi_pose import MultiPoseDataset
from .sample.rear_headlight_multi_pose import RearHeadLightMultiPoseDataset
from .dataset.coco import COCO
from .dataset.pascal import PascalVOC
from .dataset.kitti import KITTI
from .dataset.coco_hp import COCOHP
from .dataset.rear_headlight_hp import RearHeadLightHP

dataset_factory = {
    'coco': COCO,
    'pascal': PascalVOC,
    'kitti': KITTI,
    'coco_hp': COCOHP,
    'rear_headlight_hp': RearHeadLightHP
}

_sample_factory = {
    'exdet': EXDetDataset,
    'ctdet': CTDetDataset,
    'ddd': DddDataset,
    'multi_pose': MultiPoseDataset,
    'rear_headlight_multi_pose': RearHeadLightMultiPoseDataset
}


# coco + ctdet
# coco_hp + multi_pose

def get_dataset(dataset, task):
    class Dataset(dataset_factory[dataset], _sample_factory[task]):
        pass

    return Dataset
