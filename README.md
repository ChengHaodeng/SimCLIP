# SimCLIP
This repository contains the official PyTorch implementation for SimCLIP.

Anomaly Detection, consisting of anomaly classification and segmentation, has been widely applied in various applications.
Recently, Large pre-trained vision-language models, such as CLIP, have demonstrated significant potential in zero-/few-shot anomaly detection tasks.
However, existing methods not only rely on expert knowledge to manually craft extensive text prompts but also suffer from misalignment of text high-level semantic features with image patch low-level features in anomaly segmentation tasks. In this paper, we propose a SimCLIP method, which focuses on refining the aforementioned misalignment problem through Implicit Prompt Tuning (IPT) within the text feature space. In this way, our approach requires only a simple two-class prompt to accomplish anomaly classification and segmentation tasks in zero-shot scenarios efficiently. 
Furthermore, we introduce its few-shot extension, SimCLIP+, which employs normal comparison images and learned text prompts as prior knowledge to tweak the anomaly score maps. Extensive experiments on two challenging datasets prove the more remarkable generalization capacity of our method compared to the current state-of-the-art.
# Usage
### 1. Environment Setup ###
```
pip install requirements.txt
```
### 2. Prepare Dataset ###

```
data
├── mvtec
    ├── meta.json
    ├── bottle
        ├── train
            ├── good
                ├── 000.png
        ├── test
            ├── good
                ├── 000.png
            ├── anomaly1
                ├── 000.png
        ├── ground_truth
            ├── anomaly1
                ├── 000.png
```
### 3. Download Model Weights ###
* Download the weights that are pre-trained on the MVTec dataset [here](https://drive.google.com/drive/folders/1tQIySyWcKQC15Cq55Fd0gOIbzdzo-v3S?usp=drive_link).
```
pretrained_weights
├── train_on_mvtec
    ├──trainable_epoch_20.pth
    ├──clip_epoch_20.pth
```
* Download the weights that are pre-trained on the VisA dataset [here](https://drive.google.com/drive/folders/1nd5oWJdmG6_I-Ov7zJvWhRby0uRNtQaC?usp=drive_link).
```
pretrained_weights
├── train_on_visa
    ├──trainable_epoch_20.pth
    ├──clip_epoch_20.pth
```
### 4. Inference ###
After completing the above steps, you can perform inference on the SimCLIP.
* Testing on MVTec dataset
```
bash test_mvtec.sh
```
* Testing on VisA dataset
```
bash test_mvtec.sh
```




