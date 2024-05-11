import os
from typing import Union, List
from pkg_resources import packaging
import torch
import numpy as np
import torch.nn as nn

def encode_text_with_IPT(model, objs, tokenizer, device):
    def norm(embedding):
        embedding /= embedding.norm(dim=-1, keepdim=True)
        embedding = embedding.mean(dim=0)
        embedding /= embedding.norm()
        return embedding
    
    # amblation exp
    prompt_normal, prompt_abnormal = ['{} without defect.'], ['{} with defect.']
    prompt_state = [prompt_normal, prompt_abnormal]
    prompt_templates = ['a photo of a {}']
    
    text_prompts = {
        'class':{},
        'layer':{}
    }
    for obj in objs:
        # text_features = []
        cls_feats = []
        layer_feats_1 = []
        for i in range(len(prompt_state)):
            prompted_state = [state.format(obj) for state in prompt_state[i]]
            prompted_sentence = []
            for s in prompted_state:
                for template in prompt_templates:
                    prompted_sentence.append(template.format(s))
            prompted_sentence = tokenizer(prompted_sentence).to(device)
            cls_embed, layer_embed_1 = model.encode_text(prompted_sentence)
            
            cls_feats.append(norm(cls_embed))
            layer_feats_1.append(norm(layer_embed_1))

        cls_feats = torch.stack(cls_feats, dim=1).to(device)
        layer_feats_1 = torch.stack(layer_feats_1, dim=1).to(device)

        text_prompts['class'][obj] = cls_feats
        text_prompts['layer'][obj] = layer_feats_1

    return text_prompts