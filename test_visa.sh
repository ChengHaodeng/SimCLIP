# Reproduce the zero-shot and few-shot(1-shot, 2-shot and 4-shot) anomaly segmentation and classification results of SimCLIP on the VisA dataset.
# -------------------------- basic setup -----------------------------
dataset='visa'
data_path='./data/visa'
config_path='./open_clip/model_configs/ViT-L-14-336.json'
model='ViT-L-14-336'
features_list=(6 12 18 24)
few_shot_features=(6 12 18 24)
pretrained='openai'
seed=3037
image_size=518
# --------------------------------------------------------------------

# --------------------------------------------------------------------
visionAdapter_ckpt='checkpoint/train_on_mvtec/visionAdapter.pth' # path to pretrained weights
clip_ckpt='checkpoint/train_on_mvtec/clip.pth'  

zero_shot_sp='test/visa/0shot' # path to save 0-shot results
one_shot_sp='test/visa/1shot'  # path to save 1-shot results
two_shot_sp='test/visa/2hot'   # path to save 2-shot results
four_shot_sp='test/visa/4shot' # path to save 4-shot results

device='0' # device id
# --------------------------------------------------------------------

# --------------------------------------------------------------------
mode='zero_shot'
# 0-shot
python test_zero_shot.py --mode $mode --dataset $dataset --data_path $data_path --save_path $zero_shot_sp --config_path $config_path \
--model $model --features_list ${features_list[@]} --pretrained $pretrained --image_size $image_size --device $device \
--visionAdapter_ckpt $visionAdapter_ckpt --clip_ckpt $clip_ckpt --seed $seed

mode='few_shot'
# 1-shot
python test_few_shot.py --mode $mode --dataset $dataset --data_path $data_path --save_path $one_shot_sp --config_path $config_path \
--model $model --features_list ${features_list[@]} --few_shot_features ${few_shot_features[@]} --pretrained $pretrained \
--image_size $image_size --device $device --k_shot 1 --seed $seed --visionAdapter_ckpt $visionAdapter_ckpt --clip_ckpt $clip_ckpt

# 2-shot
python test_few_shot.py --mode $mode --dataset $dataset --data_path $data_path --save_path $two_shot_sp --config_path $config_path \
--model $model --features_list ${features_list[@]} --few_shot_features ${few_shot_features[@]} --pretrained $pretrained \
--image_size $image_size --device $device --k_shot 2 --seed $seed --visionAdapter_ckpt $visionAdapter_ckpt --clip_ckpt $clip_ckpt

# 4-shot
python test_few_shot.py --mode $mode --dataset $dataset --data_path $data_path --save_path $four_shot_sp --config_path $config_path \
--model $model --features_list ${features_list[@]} --few_shot_features ${few_shot_features[@]} --pretrained $pretrained \
--image_size $image_size --device $device --k_shot 4 --seed $seed --visionAdapter_ckpt $visionAdapter_ckpt --clip_ckpt $clip_ckpt