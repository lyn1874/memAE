import torch 
from torch import nn
import summary
import os
import numpy as np
import time
from torchvision import transforms
from torch.utils.data import DataLoader
import utils
from models import AutoEncoderCov3D, AutoEncoderCov3DMem
import data.utils as data_utils
import argparse
from tqdm import tqdm
import utils.eval as eval_utils

parser = argparse.ArgumentParser(description="Memorizing_Normality")
parser.add_argument('--dataset', type=str, default="UCSDped2")
parser.add_argument("--augment_type", type=str, default="original")
parser.add_argument("--version", type=int, default=1)
parser.add_argument("--ckpt_step", type=int, default=59)
parser.add_argument("--EntropyLossWeight", type=float, default=2e-4)
parser.add_argument("--lr", type=float, default=1e-4)
args = parser.parse_args()

device = "cuda"

height, width = 256, 256
ch = 1
num_frame = 16
batch_size=1
ModelName = "MemAE"
model_dir = '/project/bo/exp_data/memory_normal/%s/%slr_%.5f_entropyloss_%.5f_version_%d/' % (args.dataset, 
                                                                                              args.augment_type,
                                                                                              args.lr,
                                                                                              args.EntropyLossWeight,
                                                                                              args.version)

ckpt_dir = model_dir + "model-00%d.pt" % args.ckpt_step

data_dir = '/project/bo/anomaly_data/%s/%s/' % (args.dataset, "Test_jpg")

frame_trans = transforms.Compose([
        transforms.Resize([height, width]),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
unorm_trans = utils.UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
print("------Data folder", data_dir)
print("------Model folder", model_dir)
print("------Restored ckpt", ckpt_dir)

data_loader = data_utils.DataLoader(data_dir, frame_trans, time_step=num_frame-1, num_pred=1)
video_data_loader = DataLoader(data_loader, batch_size=batch_size, shuffle=False)

chnum_in_ = 1
mem_dim_in = 2000
sparse_shrink_thres = 0.0025

model = AutoEncoderCov3DMem(chnum_in_, mem_dim_in, shrink_thres=sparse_shrink_thres)
model_para = torch.load(ckpt_dir)
model.load_state_dict(model_para)
model.requires_grad_(False)
model.to(device)
model.eval()

img_crop_size = 0
recon_error_list = [None] * len(video_data_loader)
progress_bar = tqdm(video_data_loader)
for batch_idx, frames in enumerate(progress_bar):
    progress_bar.update()
    frames = frames.reshape([batch_size, num_frame, ch, height, width])
    frames = frames.permute(0, 2, 1, 3, 4)
    frames = frames.to(device)
    if (ModelName == 'AE'):
        recon_frames = model(frames)
        ###### calculate reconstruction error (MSE)
        recon_np = utils.vframes2imgs(unorm_trans(recon_frames.data), step=1, batch_idx=0)
        input_np = utils.vframes2imgs(unorm_trans(frames.data), step=1, batch_idx=0)
        r = utils.crop_image(recon_np, img_crop_size) - utils.crop_image(input_np, img_crop_size)
        # recon_error = np.mean(sum(r**2)**0.5)
        recon_error = np.mean(r ** 2)  # **0.5
    elif (ModelName == 'MemAE'):
        recon_res = model(frames)
        recon_frames = recon_res['output']
        recon_np = utils.vframes2imgs(unorm_trans(recon_frames.data), step=1, batch_idx=0)
        input_np = utils.vframes2imgs(unorm_trans(frames.data), step=1, batch_idx=0)
        r = utils.crop_image(recon_np, img_crop_size) - utils.crop_image(input_np, img_crop_size)
        sp_error_map = sum(r ** 2)**0.5
        recon_error = np.mean(sp_error_map.flatten())
    else:
        recon_error = -1
        print('Wrong ModelName.')
    recon_error_list[batch_idx] = recon_error

gt_file = "/project/bo/anomaly_data/%s/gt.npy" % args.dataset
print("............start to checking the anomaly detection auc score...................")
eval_utils.eval_video2(gt_file, recon_error_list, args.dataset)
# np.save(model_dir + 'recons_error_%s' % args.augment_type, recon_error_list)

