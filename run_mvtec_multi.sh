#### MVTec-AD Multiclass Anomaly Detection ####

# run.py -> Ignore for AUROC computed from run.py
data_path='/home/data/anomaly_detection/defect/'
datasets=('mvtec_all')
dataset_flags=($(for dataset in "${datasets[@]}"; do echo "${dataset}"; done))


backbone=dino
patchsize=8
gpu=7
results_path=uniformaly_results/


CUDA_LAUNCH_BLOCKING=1 python run.py mvtec $data_path -d "${dataset_flags[@]}" \
-b ${backbone}_vit_base_${patchsize} -le 2 3 4 5 6 7 8 9 --topk 0.05 --bpm --thres 0.1 --anomaly_scorer_num_nn 1 \
--gpu $gpu --num_workers 8 --faiss_on_gpu --faiss_num_workers 8 --batch_size 16 \
--log_group ${backbone}_vit_base_${patchsize} --log_project mvtec_multiclass --results_path $results_path --save_model \
--sampler approx_greedy_coreset -p 0.05 \
--seed 0  \

# eval.py
data_path='/home/data/anomaly_detection/defect/mvtec'
datasets=('screw' 'pill' 'capsule' 'cable' 'grid' 'zipper' 'transistor' 'bottle' 'carpet' 'hazelnut' 'leather' 'metal_nut' 'tile' 'toothbrush' 'wood')
dataset_flags=($(for dataset in "${datasets[@]}"; do echo "${dataset}"; done))
model_path=uniformaly_results/mvtec_multiclass/${backbone}_vit_base_${patchsize}_1/models/mvtec_mvtec_all  
results_path=uniformaly_results/mvtec_multiclass_evaluation

python eval.py mvtec $data_path -d "${dataset_flags[@]}" \
--gpu $gpu --seed 0 --faiss_on_gpu --faiss_num_workers 4 --num_workers 4 --batch_size 16 \
--topk 0.05 --thres 0.1 \
--results_path $results_path --model_path $model_path \
--resize 256 --imagesize 224 \