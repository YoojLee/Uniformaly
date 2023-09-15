#### Species60 ####
data_path='/home/data/anomaly_detection/semantic/species60'
count=30 
min=0
max=59

split_seed=218  
random_numbers=$(shuf -i $min-$max -n $count)

datasets=$(shuf -i $min-$max -n $count --random-source=<(openssl enc -aes-256-ctr -pass pass:"$split_seed" -nosalt < /dev/zero 2>/dev/null))
dataset_flags=($(for dataset in "${datasets[@]}"; do echo "${dataset}"; done))

gpu=0
arch=ibot
patchsize=16

CUDA_LAUNCH_BLOCKING=1 python run.py species $data_path -d "${dataset_flags[@]}" --use_multiclass \
--resize 256 --imagesize 224 \
-b ${arch}_vit_base_${patchsize} -le 2 3 4 5 6 7 8 9 --topk 0.05 --anomaly_scorer_num_nn 1 --thres 0.1 --bpm \
--gpu $gpu --num_workers 4 --faiss_on_gpu --faiss_num_workers 4 --batch_size 1 \
--log_group ${arch}_vit_base_${patchsize} --log_project species_multi --save_model \
--sampler approx_greedy_coreset -p 0.01 \
--seed 0 
