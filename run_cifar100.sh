#### Cifar-100 ####
data_path='/home/data/anomaly_detection/semantic/cifar100'

dataset_flags=($(for dataset in {0..19}; do echo "${dataset}"; done))
gpu=0
arch=ibot
patchsize=16
seed=0

CUDA_LAUNCH_BLOCKING=1 python run.py cifar100 $data_path -d "${dataset_flags[@]}" \
--resize 112 --imagesize 224 \
-b ${arch}_vit_base_${patchsize} -le 2 3 4 5 6 7 8 9 --topk 0.05 --anomaly_scorer_num_nn 1 --thres 0.1 --bpm \
--gpu $gpu --num_workers 16 --faiss_on_gpu --faiss_num_workers 16 --batch_size 1 \
--log_group ${arch}_vit_base_${patchsize} --log_project cifar100_one --save_model \
--sampler approx_greedy_coreset -p 0.05 --seed $seed \