#### Cifar-10 ####
data_path='/home/data/anomaly_detection/semantic/cifar10'
datasets=('5 6 7 8 9') # ('0 1 2 3 4') ('0 2 4 6 8') ('1 3 5 7 9')

dataset_flags=($(for dataset in "${datasets[@]}"; do echo "${dataset}"; done))
gpu=0
arch=ibot
patchsize=16
seed=218

CUDA_LAUNCH_BLOCKING=1 python run.py cifar10 $data_path -d "${dataset_flags[@]}" --use_multiclass \
--resize 112 --imagesize 224 \
-b ${arch}_vit_base_${patchsize} -le 2 3 4 5 6 7 8 9 --topk 0.05 --anomaly_scorer_num_nn 1 --bpm --thres 0.1 \
--gpu $gpu --num_workers 4 --faiss_on_gpu --faiss_num_workers 4 --batch_size 1 \
--log_group ${arch}_vit_base_${patchsize} --log_project cifar_multi --save_model \
--sampler approx_greedy_coreset -p 0.05 \
--seed $seed \
