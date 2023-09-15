#### ImageNet-30 ####
data_path='/home/data/anomaly_detection/semantic/imagenet30'

dataset_flags=($(for dataset in {0..29}; do echo "${dataset}"; done))
gpu=0
arch=dino
patchsize=8

CUDA_LAUNCH_BLOCKING=1 python run.py imagenet $data_path -d "${dataset_flags[@]}" \
-b ${arch}_vit_base_${patchsize} -le 2 3 4 5 6 7 8 9 --topk 0.05 --anomaly_scorer_num_nn 1 --thres 0.1 --bpm \
--gpu $gpu --num_workers 4 --faiss_on_gpu --faiss_num_workers 4 --batch_size 1 \
--log_group ${arch}_vit_base_${patchsize} --log_project in30_one --save_model \
--sampler approx_greedy_coreset -p 0.01 \
--seed 0 
