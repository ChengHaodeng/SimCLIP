import torch
from utils.dataset import VisaDataset, MVTecDataset
import numpy as np

def memory(model_name, model, obj_list, dataset_dir, save_path, preprocess, transform, k_shot,
           few_shot_features, dataset_name, device):
    mem_features = {}
    for obj in obj_list:
        if dataset_name == 'mvtec':
            data = MVTecDataset(root=dataset_dir, transform=preprocess, target_transform=transform,
                                aug_rate=-1, mode='train', k_shot=k_shot, save_dir=save_path, obj_name=obj)
        else:
            data = VisaDataset(root=dataset_dir, transform=preprocess, target_transform=transform,
                               mode='train', k_shot=k_shot, save_dir=save_path, obj_name=obj)
        dataloader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False)
        features = []
        for items in dataloader:
            image = items['img'].to(device)
            with torch.no_grad():
                image_features, patch_tokens = model.encode_image(image, few_shot_features)
                if 'ViT' in model_name:
                    patch_tokens = [p[0, 1:, :] for p in patch_tokens]
                else:
                    patch_tokens = [p[0].view(p.shape[1], -1).permute(1, 0).contiguous() for p in patch_tokens]
                features.append(patch_tokens)
        mem_features[obj] = [torch.cat(
            [features[j][i] for j in range(len(features))], dim=0) for i in range(len(features[0]))]
    return mem_features

def get_normal_scoremap(cls_name, patch_tokens, linearlayer, text_prompts):

    feats = text_prompts['layer'][cls_name]

    patch_tokens = linearlayer(patch_tokens)
    anomaly_maps = []
    for layer in range(len(patch_tokens)):
        patch_tokens[layer] /= patch_tokens[layer].norm(dim=-1, keepdim=True)
        anomaly_map = (100.0 * patch_tokens[layer] @ feats.to(patch_tokens[layer].dtype))
        B, L, C = anomaly_map.shape
        H = int(np.sqrt(L))
        anomaly_map = torch.softmax(anomaly_map.permute(0, 2, 1).view(B, 2, H, H), dim=1)[:, 1, :, :]
        anomaly_maps.append(anomaly_map.cpu().numpy())

    anomaly_maps = np.sum(anomaly_maps, axis=0)
    return anomaly_maps