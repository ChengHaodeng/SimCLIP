import os
import cv2
import json
import torch
import random
import logging
import argparse
import numpy as np
from PIL import Image
from skimage import measure
from tabulate import tabulate
import torch.nn.functional as F
import torchvision.transforms as transforms
from sklearn.metrics import auc, roc_auc_score, average_precision_score, f1_score, precision_recall_curve, pairwise

import open_clip
# from utils.few_shot import memory
# from utils import LinearLayer
# from utils import VisaDataset, MVTecDataset
from utils import *

from tqdm import tqdm
from scipy import ndimage
from matplotlib import pyplot as plt

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize(pred, max_value=None, min_value=None):
    if max_value is None or min_value is None:
        return (pred - pred.min()) / (pred.max() - pred.min())
    else:
        return (pred - min_value) / (max_value - min_value)


def apply_ad_scoremap(image, scoremap, alpha=0.5):
    np_image = np.asarray(image, dtype=float)
    scoremap = (scoremap * 255).astype(np.uint8)
    color_map = np.zeros((518, 518, 3), dtype=np.uint8)
    color_map[..., 0] = scoremap
    return (alpha * np_image + (1 - alpha) * color_map).astype(np.uint8)


def cal_pro_score(masks, amaps, max_step=200, expect_fpr=0.3):
    # ref: https://github.com/gudovskiy/cflow-ad/blob/master/train.py
    binary_amaps = np.zeros_like(amaps, dtype=bool)
    min_th, max_th = amaps.min(), amaps.max()
    delta = (max_th - min_th) / max_step
    pros, fprs, ths = [], [], []
    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th], binary_amaps[amaps > th] = 0, 1
        pro = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                tp_pixels = binary_amap[region.coords[:, 0], region.coords[:, 1]].sum()
                pro.append(tp_pixels / region.area)
        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()
        pros.append(np.array(pro).mean())
        fprs.append(fpr)
        ths.append(th)
    pros, fprs, ths = np.array(pros), np.array(fprs), np.array(ths)
    idxes = fprs < expect_fpr
    fprs = fprs[idxes]
    fprs = (fprs - fprs.min()) / (fprs.max() - fprs.min())
    pro_auc = auc(fprs, pros[idxes])
    return pro_auc


def test(args):
    img_size = args.image_size
    features_list = args.features_list
    dataset_dir = args.data_path
    save_path = args.save_path
    dataset_name = args.dataset
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    device = torch.device(f'cuda:{args.device}')
    txt_path = os.path.join(save_path, 'log.txt')
    print(f"device:{device}")
    # clip
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, img_size, pretrained=args.pretrained)
    clip_model = torch.load(args.clip_path, map_location=device)
    model.load_state_dict(clip_model['clip'], strict=True)
    model.to(device)
    tokenizer = open_clip.get_tokenizer(args.model)

    # logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.setLevel(logging.WARNING)
    logger = logging.getLogger('test')
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
        if args.mode == 'zero_shot' and (arg == 'k_shot' or arg == 'few_shot_features'):
            continue
        logger.info(f'{arg}: {getattr(args, arg)}')

    # seg
    with open(args.config_path, 'r') as f:
        model_configs = json.load(f)
    linearlayer = LinearLayer(model_configs['vision_cfg']['width'], model_configs['embed_dim'],
                              len(features_list), args.model).to(device)
    checkpoint = torch.load(args.checkpoint_path, map_location = device)
    linearlayer.load_state_dict(checkpoint["trainable_linearlayer"])

    # dataset
    transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor()
        ])
    if dataset_name == 'mvtec':
        test_data = MVTecDataset(root='data/mvtec', transform=preprocess, target_transform=transform, aug_rate=-1, mode='test')
        print("Testing on mvtec dataset......")
    elif dataset_name == 'visa':
        test_data = VisaDataset(root='data/visa', transform=preprocess, target_transform=transform, mode='test')
        print("Testing on visa dataset......")

    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    obj_list = test_data.get_cls_names()

    # text prompt
    with torch.cuda.amp.autocast(), torch.no_grad():
        text_prompts = encode_text_with_multi_learn(model, obj_list, tokenizer, device)

    results = {}
    results['cls_names'] = []
    results['imgs_masks'] = []
    results['anomaly_maps'] = []
    results['gt_sp'] = []
    results['pr_sp'] = []
    vis_ls = []
    for idx, items in enumerate(tqdm(test_dataloader)):
        image = items['img'].to(device)
        cls_name = items['cls_name']
        results['cls_names'].append(cls_name[0])
        gt_mask = items['img_mask']
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0

        results['imgs_masks'].append(gt_mask)  # px
        results['gt_sp'].append(items['anomaly'].item())

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features, patch_tokens = model.encode_image(image, features_list)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            cls_features = []
            for cls in cls_name:
                cls_features.append(text_prompts['class'][cls])
            cls_features = torch.stack(cls_features, dim=0)

            # sample
            text_probs = (100.0 * image_features @ cls_features[0]).softmax(dim=-1)
            results['pr_sp'].append(text_probs[0][1].cpu().item())

            feats = []
            for cls in cls_name:
                feats.append(text_prompts['layer'][cls])
            feats = torch.stack(feats, dim=0) # [bsz, 768, 2]

            # pixel
            patch_tokens = linearlayer(patch_tokens)
            anomaly_maps = []
            for layer in range(len(patch_tokens)):
                cur_layer_text_features = feats
                patch_tokens[layer] /= patch_tokens[layer].norm(dim=-1, keepdim=True)
                anomaly_map = (100.0 * patch_tokens[layer] @ cur_layer_text_features)
                B, L, C = anomaly_map.shape
                H = int(np.sqrt(L))
                anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                                            size=img_size, mode='bilinear', align_corners=True)
                anomaly_map = torch.softmax(anomaly_map, dim=1)[:, 1, :, :]
                anomaly_map = ndimage.gaussian_filter(anomaly_map.cpu().numpy(), sigma=7)
                anomaly_maps.append(anomaly_map)
            anomaly_map = np.sum(anomaly_maps, axis=0)
            results['anomaly_maps'].append(anomaly_map)

    # metrics
    table_ls = []
    auroc_sp_ls = []
    auroc_px_ls = []
    f1_sp_ls = []
    f1_px_ls = []
    aupro_ls = []
    ap_sp_ls = []
    ap_px_ls = []
    min_max_px = {} # saves the maximum and minimum values of the pixel level
    thresholds_px = {} # threshold for visualization
    for obj in obj_list:
        table = []
        gt_px = []
        pr_px = []
        gt_sp = []
        pr_sp = []
        pr_sp_tmp = []
        table.append(obj)
        for idxes in range(len(results['cls_names'])):
            if results['cls_names'][idxes] == obj:
                gt_px.append(results['imgs_masks'][idxes].squeeze(1).numpy())
                pr_px.append(results['anomaly_maps'][idxes])
                pr_sp_tmp.append(np.max(results['anomaly_maps'][idxes]))
                gt_sp.append(results['gt_sp'][idxes])
                pr_sp.append(results['pr_sp'][idxes])
        gt_px = np.array(gt_px)
        gt_sp = np.array(gt_sp)
        pr_px = np.array(pr_px)
        pr_sp = np.array(pr_sp)
        if args.mode == 'few_shot':
            pr_sp_tmp = np.array(pr_sp_tmp)
            pr_sp_tmp = (pr_sp_tmp - pr_sp_tmp.min()) / (pr_sp_tmp.max() - pr_sp_tmp.min())
            pr_sp = 0.5 * (pr_sp + pr_sp_tmp)

        auroc_px = roc_auc_score(gt_px.ravel(), pr_px.ravel())
        auroc_sp = roc_auc_score(gt_sp, pr_sp)
        ap_sp = average_precision_score(gt_sp, pr_sp)
        ap_px = average_precision_score(gt_px.ravel(), pr_px.ravel())
        # f1_sp
        precisions, recalls, thresholds = precision_recall_curve(gt_sp, pr_sp)
        f1_scores = (2 * precisions * recalls) / (precisions + recalls)
        f1_sp = np.max(f1_scores[np.isfinite(f1_scores)])
        # f1_px
        precisions, recalls, thresholds = precision_recall_curve(gt_px.ravel(), pr_px.ravel())
        f1_scores = (2 * precisions * recalls) / (precisions + recalls)
        f1_px = np.max(f1_scores[np.isfinite(f1_scores)])

        thresholds_px[obj] = thresholds[np.argmax(f1_scores)]
        # aupro
        if len(gt_px.shape) == 4:
            gt_px = gt_px.squeeze(1)
        if len(pr_px.shape) == 4:
            pr_px = pr_px.squeeze(1)
        aupro = cal_pro_score(gt_px, pr_px)

        table.append(str(np.round(auroc_px * 100, decimals=1)))
        table.append(str(np.round(f1_px * 100, decimals=1)))
        table.append(str(np.round(ap_px * 100, decimals=1)))
        table.append(str(np.round(aupro * 100, decimals=1)))
        table.append(str(np.round(auroc_sp * 100, decimals=1)))
        table.append(str(np.round(f1_sp * 100, decimals=1)))
        table.append(str(np.round(ap_sp * 100, decimals=1)))

        table_ls.append(table)
        auroc_sp_ls.append(auroc_sp)
        auroc_px_ls.append(auroc_px)
        f1_sp_ls.append(f1_sp)
        f1_px_ls.append(f1_px)
        aupro_ls.append(aupro)
        ap_sp_ls.append(ap_sp)
        ap_px_ls.append(ap_px)
        print(f"cls:{obj}\t I-AUROC:{auroc_sp}\t P-AUROC:{auroc_px} \n")

    # logger
    table_ls.append(['mean', str(np.round(np.mean(auroc_px_ls) * 100, decimals=1)),
                     str(np.round(np.mean(f1_px_ls) * 100, decimals=1)), str(np.round(np.mean(ap_px_ls) * 100, decimals=1)),
                     str(np.round(np.mean(aupro_ls) * 100, decimals=1)), str(np.round(np.mean(auroc_sp_ls) * 100, decimals=1)),
                     str(np.round(np.mean(f1_sp_ls) * 100, decimals=1)), str(np.round(np.mean(ap_sp_ls) * 100, decimals=1))])
    results = tabulate(table_ls, headers=['objects', 'auroc_px', 'f1_px', 'ap_px', 'aupro', 'auroc_sp',
                                          'f1_sp', 'ap_sp'], tablefmt="pipe")
    

    logger.info(results)
    with open('result.txt', 'a+') as f:
        for arg in vars(args):
            f.write(f"{arg}:{getattr(args, arg)}\n")
        f.write(results)
        f.write('\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser("SimCLIP", add_help=True)
    # paths
    parser.add_argument("--data_path", type=str, default="./data/mvtec", help="path to test dataset")
    parser.add_argument("--save_path", type=str, default='./results/ablation_mvtec_0_shot_gt', help='path to save results')
    parser.add_argument("--checkpoint_path", type=str, default='', help='path to save results')
    parser.add_argument("--config_path", type=str, default='open_clip/model_configs/ViT-L-14-336.json', help="model configs")
    # model
    parser.add_argument("--dataset", type=str, default='mvtec', help="test dataset")
    parser.add_argument("--model", type=str, default="ViT-L-14-336", help="model used")
    parser.add_argument("--pretrained", type=str, default="openai", help="pretrained weight used")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--mode", type=str, default="zero_shot", help="")
    parser.add_argument("--seed", type=int, default=1000, help="random seed")
    parser.add_argument("--clip_path", type=str, default="")
    parser.add_argument("--device", type=str, default='0')
    args = parser.parse_args()

    setup_seed(args.seed)
    test(args)
