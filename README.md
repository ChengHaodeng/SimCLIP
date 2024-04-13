# SimCLIP
This repository contains the official PyTorch implementation for SimCLIP.

Recently, Large pre-trained vision-language models, such as CLIP, have demonstrated significant potential in zero-/few-shot anomaly detection tasks.
However, existing methods not only rely on expert knowledge to manually craft extensive text prompts but also suffer from a misalignment of high-level language features with fine-level vision features in anomaly segmentation tasks. In this paper, we propose a method, named SimCLIP, which focuses on refining the aforementioned misalignment problem through bidirectional adaptation of both Multi-Hierarchy Vision Adapter (MHVA) and Implicit Prompt Tuning (IPT). In this way, our approach requires only a simple binary prompt to accomplish anomaly classification and segmentation tasks in zero-shot scenarios efficiently. Furthermore, we introduce its few-shot extension, SimCLIP+, integrating the relational information among vision embedding and skillfully merging the cross-modal synergy information between vision and language to address AD tasks. Extensive experiments on two challenging datasets prove the more remarkable generalization capacity of our method compared to the current state-of-the-art.
### 1. Environment Setup ###
```
pip install requirements.txt
```
### 2. Prepare Dataset ###
#### MVTec AD 
* Download the [MVTec](https://www.mvtec.com/company/research/datasets/mvtec-ad) dataset and place it into  ```data/mvtec```
* Run```python data/mvtec.py```to generate the ```meta.json```file.
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
#### VisA 
* Download the [VisA](https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar) dataset and place it into  ```data/visa```
* Run```python data/visa.py```to generate the ```meta.json```file.
```
data
├── visa
    ├── meta.json
    ├── candle
        ├── Data
            ├── Images
                ├── Anomaly
                    ├── 000.JPG
                ├── Normal
                    ├── 0000.JPG
            ├── Masks
                ├── Anomaly
                    ├── 000.png
```

### 3. Download Model Weights ###
* Download the [weights](https://mega.nz/folder/lLUGHCbB#qnTEmwxeNiaTI28XUXNYdw) that are pre-trained on the MVTec dataset.
* Move the weights to the ```pretrain_weights/train_on_mvtec```
```
pretrain_weights
├── train_on_mvtec
    ├──trainable_epoch_20.pth
    ├──clip_epoch_20.pth
```
* Download the [weights](https://mega.nz/folder/UC92XY6Y#O-oIXszGKYonFuTyXL1DDw) that are pre-trained on the VisA dataset.
* Move the weights to the ```pretrain_weights/train_on_visa```
```
pretrain_weights
├── train_on_visa
    ├──trainable_epoch_20.pth
    ├──clip_epoch_20.pth
```
### 4. Inference ###
After completing the above steps, you can perform inference on the SimCLIP.
* Inference on MVTec dataset (Both zero-shot and few-shot)
```
bash test_mvtec.sh
```


* Inference on VisA dataset (Both zero-shot and few-shot)
```
bash test_mvtec.sh
```




