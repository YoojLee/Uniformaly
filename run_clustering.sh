data_path='/home/data/anomaly_detection/defect/mvtec'
datasets=('screw' 'pill' 'capsule' 'cable' 'grid' 'zipper' 'transistor' 'bottle' 'carpet' 'hazelnut' 'leather' 'metal_nut' 'tile' 'toothbrush' 'wood')

dataset_flags=($(for dataset in "${datasets[@]}"; do echo "${dataset}"; done))
gpu=0
arch=dino
patchsize=8
nn=1
results_path=uniformaly_results/clustering/


CUDA_LAUNCH_BLOCKING=1 python run.py mvtec $data_path -d "${dataset_flags[@]}" \
-b ${arch}_vit_base_${patchsize} -le 2 3 4 5 6 7 8 9 --topk 0.05 --anomaly_scorer_num_nn $nn --thres 0.1 --bpm --return_topk_index \
--gpu $gpu --num_workers 4 --faiss_on_gpu --faiss_num_workers 4 --batch_size 1 \
--log_group ${arch}_vit_base_${patchsize} --log_project clustering --results_path $results_path \
--sampler approx_greedy_coreset -p 0.1 --patchsize 7 \
--seed 0 
