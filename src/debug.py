python  datasets/sample/rear_headlight_multi_pose.py  rear_headlight_multi_pose --not_rand_crop --no_color_aug --debug 1

self.parser.add_argument('--arch', default='dla_34',
                         help='model architecture. Currently tested'
                              'res_18 | res_101 | resdcn_18 | resdcn_101 |'

# train
self.parser.add_argument('--lr', type=float, default=1.25e-4,
                         help='learning rate for batch size 32.')
self.parser.add_argument('--lr_step', type=str, default='90,120',
                         help='drop learning rate by 10.')
self.parser.add_argument('--num_epochs', type=int, default=140,
                         help='total training epochs.')
self.parser.add_argument('--batch_size', type=int, default=32,
                         help='batch size')
self.parser.add_argument('--master_batch_size', type=int, default=-1,
                         help='batch size on the master gpu.')
self.parser.add_argument('--num_iters', type=int, default=-1,
                         help='default: #samples / batch_size.')
self.parser.add_argument('--val_intervals', type=int, default=5,
                         help='number of epochs to run validation.')
self.parser.add_argument('--trainval', action='store_true',
                         help='include validation in training and '
                              'test on test set')


python main.py --dataset rear_headlight_hp rear_headlight_multi_pose --not_rand_crop --no_color_aug --exp_id rear_headlight --arch res_18 --batch_size 1 --gpus 0