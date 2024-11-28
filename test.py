import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
import imageio
import time
import yaml
from model.SOLNet import SOLNet
from data import test_dataset


torch.cuda.set_device(0)
parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=256, help='testing size')
opt = parser.parse_args()

dataset_path = './dataset/'

model = SOLNet(deploy=True)
# model.load_state_dict(torch.load('model_path'))
model.eval()
model.cuda()

test_datasets = ['ORSSD', 'EORSSD']

for dataset in test_datasets:
    save_path = './results/xxx/prediction/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = dataset_path + dataset + '/test/images/'
    print(dataset)
    gt_root = dataset_path + dataset + '/test/GT/'
    test_loader = test_dataset(image_root, gt_root, opt.testsize)
    time_sum = 0
    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        time_start = time.time()
        feature_1, feature_2, feature_3, feature_4, feature_1_sig, feature_2_sig, feature_3_sig, feature_4_sig = model(image)
        time_end = time.time()
        time_sum = time_sum+(time_end-time_start)
        feature_4 = F.interpolate(feature_4, size=gt.shape, mode='bilinear', align_corners=False)
        feature_4 = feature_4.sigmoid().data.cpu().numpy().squeeze()
        feature_4 = (feature_4 - feature_4.min()) / (feature_4.max() - feature_4.min() + 1e-8)
        feature_4 = (feature_4 *  255.0).astype(np.uint8)
        imageio.imsave(save_path+name, feature_4)

    print('Running time {:.5f}'.format(time_sum/test_loader.size))
    print('FPS {:.5f}'.format(test_loader.size / time_sum))
