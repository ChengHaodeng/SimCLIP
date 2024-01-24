import torch
import torch.nn as nn
import numpy as np
import random
import os
import json
import argparse
from torch.utils.data import DataLoader
from datetime import datetime
from torch.nn import functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import logging
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, MultiStepLR
import open_clip
from dataset import VisaDataset, MVTecDataset
from model import LinearLayer
from loss import FocalLoss, BinaryDiceLoss
from prompt_ensemble import *
import sys
from tqdm import tqdm

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(args):
    # configs
    epochs = args.epoch
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    image_size = args.image_size
    device = torch.device(f'cuda:{args.device}')
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    txt_path = os.path.join(save_path, 'log.txt')  # log

    # model configs
    features_list = args.features_list
    with open(args.config_path, 'r') as f:
        model_configs = json.load(f)

    # clip model
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, image_size, pretrained=args.pretrained)
    model.to(device)
    tokenizer = open_clip.get_tokenizer(args.model)
    
    # loading checkpoint model for Clip model
    if args.clip_checkpoint is not None:
        print(f'loading Clip model for {args.clip_checkpoint}')
        clip_checkpoint = torch.load(args.clip_checkpoint)
        model.load_state_dict(clip_checkpoint['clip'])

    # logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.setLevel(logging.WARNING)
    logger = logging.getLogger('train')
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(txt_path, mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # record parameters
    for arg in vars(args):
        logger.info(f'{arg}: {getattr(args, arg)}')

    # transforms
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor()
    ])
    
    # datasets
    if args.dataset == 'mvtec':
        train_data = MVTecDataset(root=args.train_data_path, transform=preprocess, target_transform=transform, aug_rate=args.aug_rate)
    else:
        train_data = VisaDataset(root=args.train_data_path, transform=preprocess, target_transform=transform)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # linear layer
    trainable_layer = LinearLayer(model_configs['vision_cfg']['width'], model_configs['embed_dim'],
                                  len(args.features_list), args.model).to(device)
    
    # loading checkpoint model for trainable_layer
    if args.trainable_checkpoint is not None:
        print(f'loading trainable_layer for {args.trainable_checkpoint}')
        trainable_checkpoint = torch.load(args.trainable_checkpoint)
        trainable_layer.load_state_dict(trainable_checkpoint["trainable_linearlayer"])

    optim_params = []
    for name, params in model.named_parameters():
        if 'learner' in name or 'text_projection' in name or 'visual.proj' in name:
            optim_params.append(params)
            params.requires_grad_(True)
            print(f"{name}.requires_grad={True}")
        else:
            params.requires_grad_(False)
    optim_params.extend(list(trainable_layer.parameters()))
    optimizer = torch.optim.Adam(optim_params, lr=0.001, betas=(0.5, 0.999))
    scheduler = CosineAnnealingLR(optimizer, T_max = int(epochs * (1.2)))
    
    # losses
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()

    for epoch in range(epochs):
        loss_list = []
        idx = 0
        
        for items in train_dataloader:
            idx += 1
            image = items['img'].to(device)
            cls_name = items['cls_name']
            aux_gt = items['aux_img_mask'].squeeze().to(device)
            aux_gt[aux_gt > 0.5], aux_gt[aux_gt <= 0.5] = 1, 0
            target = items['anomaly'].to(device)
            with torch.cuda.amp.autocast():
                obj_list = train_data.get_cls_names()
                text_prompts = encode_text_with_multi_learn(model, obj_list, tokenizer, device)
                
                image_features, patch_tokens = model.encode_image(image, features_list)

                cls_features = []
                for cls in cls_name:
                    cls_features.append(text_prompts['class'][cls])
                cls_features = torch.stack(cls_features, dim=0) # [bsz, 768, 2]

                feats = []
                for cls in cls_name:
                    feats.append(text_prompts['layer'][cls])
                feats = torch.stack(feats, dim=0) # [bsz, 768, 2]

                # pixel level
                patch_tokens = trainable_layer(patch_tokens)
                anomaly_maps = []
                anomaly_maps_sm = []
                for layer in range(len(patch_tokens)):
                    cur_layer_text_features = feats
                    patch_tokens[layer] /= patch_tokens[layer].norm(dim=-1, keepdim=True)
                    anomaly_map = (100.0 * patch_tokens[layer] @ cur_layer_text_features)
                    B, L, C = anomaly_map.shape
                    H = int(np.sqrt(L))
                    # get small anomaly_maps
                    anomaly_maps_sm.append(torch.softmax(anomaly_map.permute(0, 2, 1).view(B, 2, H, H), dim=1))
                    # get big anomaly_maps
                    anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                                                size=image_size, mode='bilinear', align_corners=True)
                    anomaly_map = torch.softmax(anomaly_map, dim=1)
                    anomaly_maps.append(anomaly_map)
                # image level
                image_features /= image_features.norm(dim=-1, keepdim=True) # [bsz, 768]
                image_features = image_features.unsqueeze(dim=1)
                anomaly_probs = (100.0 * image_features @ cls_features).squeeze(dim=1)

            # losses
            gt = items['img_mask'].squeeze().to(device)
            gt[gt > 0.5], gt[gt <= 0.5] = 1, 0
            # classification loss
            cls_loss = F.cross_entropy(anomaly_probs, target)
            # segmentation loss
            seg_loss = 0
            for num in range(len(anomaly_maps)):
                seg_loss += loss_focal(anomaly_maps[num], gt)
                seg_loss += loss_dice(anomaly_maps[num][:, 1, :, :], gt)
            
            hyperparam_lambda, hyperparam_gamma = 1.3, 0.08
            loss = hyperparam_lambda * cls_loss + hyperparam_gamma * seg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())

        scheduler.step()
        # logs
        if (epoch + 1) % args.print_freq == 0:
            logger.info('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epochs, np.mean(loss_list)))

        # save model
        if (epoch + 1) % args.save_freq == 0:
            ckp_path = os.path.join(save_path, 'trainable_epoch_' + str(epoch + 1) + '.pth')
            torch.save({'trainable_linearlayer': trainable_layer.state_dict()}, ckp_path)
            print(f"saving {ckp_path} now...")

        if (epoch + 1) % args.save_freq == 0:
            ckp_path = os.path.join(save_path, 'clip_epoch_' + str(epoch + 1) + '.pth')
            torch.save({'clip': model.state_dict()}, ckp_path)
            print(f"saving {ckp_path} now...")

if __name__ == '__main__':
    parser = argparse.ArgumentParser("VAND Challenge", add_help=True)
    # path
    parser.add_argument("--train_data_path", type=str, default="./data/mvtec", help="train dataset path")
    parser.add_argument("--save_path", type=str, default='./exps/learnable_prompt', help='path to save results')
    parser.add_argument("--config_path", type=str, default='./open_clip/model_configs/ViT-L-14-336.json', help="model configs")
    # model
    parser.add_argument("--dataset", type=str, default='mvtec', help="train dataset name")
    parser.add_argument("--model", type=str, default="ViT-L-14-336", help="model used")
    parser.add_argument("--pretrained", type=str, default="openai", help="pretrained weight used")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")
    # hyper-parameter
    parser.add_argument("--epoch", type=int, default=3, help="epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--aug_rate", type=float, default=0.2, help="image size")
    parser.add_argument("--print_freq", type=int, default=1, help="print frequency")
    parser.add_argument("--save_freq", type=int, default=1, help="save frequency")
    parser.add_argument("--device", type=str, default='3')

    parser.add_argument("--trainable_checkpoint", type=str, default=None)
    parser.add_argument("--clip_checkpoint", type=str, default=None)
    args = parser.parse_args()
    setup_seed(111)
    train(args)

