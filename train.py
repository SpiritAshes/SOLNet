import torch
from torch.autograd import Variable
import numpy as np
import os
from tqdm import tqdm
from datetime import datetime
from model.SOLNet import SOLNet
from data import get_loader
from utils import adjust_lr, load_config
from loss import loss_func

from torch.utils.tensorboard import SummaryWriter


config = load_config('SOLNet.yaml')


image_root = config['input_path'] + 'train/images/'
gt_root = config['input_path'] + 'train/GT/'
train_loader = get_loader(image_root, gt_root, batchsize=config['batchsize'], trainsize=config['trainsize'])
total_step = len(train_loader)

model = SOLNet()
model.cuda()
params = model.parameters()
optimizer = torch.optim.Adam(params, config['lr'])

col_names = ['Loss', 'Loss4', 'Precision', 'Recall', 'Fbeta', 'MAE', 'Smearsure']

if not os.path.exists(config['save_path']):
    os.makedirs(config['save_path'])

with open(config['save_path'] + 'train_loss.csv', 'a') as f:
    np.savetxt(f, [col_names], delimiter=',', newline='\n', comments='', fmt='%s')


if not os.path.exists(config['logdir']):
    os.makedirs(config['logdir'])
writer = SummaryWriter(config['logdir'])

print("Start!")
for epoch in range(config['epoch']):

    loss4_mean = 0.0
    loss_mean = 0.0

    train_Fbeta_mean = 0.0
    train_MAE_mean = 0.0
    train_precision_mean = 0.0
    train_recall_mean = 0.0
    train_Smearsure_mean = 0.0

    adjust_lr(optimizer, config['lr'], epoch, config['decay_rate'], config['decay_epoch'])

    model.train()
    for i, pack in tqdm(enumerate(train_loader, start=0), desc='Train Step', total=total_step, leave=False, position=1):
        optimizer.zero_grad()
        images, gts = pack
        images = Variable(images)
        gts = Variable(gts)
        images = images.cuda()
        gts = gts.cuda()

        feature_1, feature_2, feature_3, feature_4, feature_1_sig, feature_2_sig, feature_3_sig, feature_4_sig = model(images)

        loss, loss4, train_precision_4, train_recall_4, train_Fbeta_4, train_MAE, train_Smearsure = loss_func(feature_1, feature_2, feature_3, feature_4, feature_1_sig, feature_2_sig, feature_3_sig, feature_4_sig, gts)

        loss.backward()
        optimizer.step()

        loss4_mean += loss4.item()
        loss_mean += loss.item()

        train_precision_mean += train_precision_4.item()
        train_recall_mean += train_recall_4.item()
        train_Fbeta_mean += train_Fbeta_4.item()
        train_MAE_mean += train_MAE.item()
        train_Smearsure_mean += train_Smearsure.item()

        print('\r{} Epoch [{:03d}/{:03d}], Step [{:05d}/{:05d}], Loss: {:.4f}, Loss4: {:.4f}, Train_P: {:.4f}, Train_R: {:.4f}, Train_Fbeta: {:.4f}, train_MAE: {:.4f}, train_Smearsure: {:.4f}'.
                format(datetime.now().strftime('%Y-%m-%d %H:%M'), epoch+1, config['epoch'], i+1, total_step, loss.item(), loss4.item(), train_precision_4.item(), train_recall_4.item(), train_Fbeta_4.item(), train_MAE.item(), train_Smearsure.item()), end='')

    loss4_mean = loss4_mean / total_step
    loss_mean = loss_mean / total_step

    train_precision_mean = train_precision_mean / total_step
    train_recall_mean = train_recall_mean / total_step
    train_Fbeta_mean = train_Fbeta_mean / total_step
    train_MAE_mean = train_MAE_mean / total_step
    train_Smearsure_mean = train_Smearsure_mean / total_step

    loss_write = np.column_stack((loss_mean, loss4_mean, train_precision_mean, train_recall_mean, train_Fbeta_mean, train_MAE_mean, train_Smearsure_mean))

    writer.add_scalar('train_loss_mean', loss_mean, epoch)
    writer.add_scalar('train_loss4_mean', loss4_mean, epoch)
    writer.add_scalar('train_precision_mean', train_precision_mean, epoch)
    writer.add_scalar('train_recall_mean', train_recall_mean, epoch)
    writer.add_scalar('train_Fbeta_mean', train_Fbeta_mean, epoch)
    writer.add_scalar('train_MAE_mean', train_MAE_mean, epoch)
    writer.add_scalar('train_Smearsure_mean', train_Smearsure_mean, epoch)

    print('\nTrain: Epoch [{:03d}/{:03d}], Step [{:05d}/{:05d}], Loss: {:.4f}, Loss4: {:.4f}, Train_P_mean: {:.4f}, Train_R_mean: {:.4f}, Train_Fbeta_mean: {:.4f}, train_MAE_mean: {:.4f}, train_Smearsure_mean: {:.4f}'.
            format(epoch+1, config['epoch'], i+1, total_step, loss_mean, loss4_mean, train_precision_mean, train_recall_mean, train_Fbeta_mean, train_MAE_mean, train_Smearsure_mean))

    with open(config['save_path'] + 'train_loss.csv', 'a') as f:
        np.savetxt(f, loss_write, delimiter=',', newline='\n', comments='', fmt='%.4f')

    torch.save(model.state_dict(), config['save_path'] + 'SOLNet.pth'+ '.%d' % (epoch+1), _use_new_zipfile_serialization=False)

