import contextlib
import logging
import os
import sys

import argparse
import numpy as np
import torch

import uniformaly.encoder
import uniformaly.cluster
import uniformaly.common
import uniformaly.metrics
import uniformaly.uniformaly
import uniformaly.sampler
import uniformaly.utils

import wandb

LOGGER = logging.getLogger(__name__)

_DATASETS = {"mvtec": ["datasets.mvtec", "MVTecDataset"],
            "species": ["datasets.species", "SpeciesDataset"],
            "imagenet": ["datasets.species", "SpeciesDataset"],
            "cifar10": ["datasets.cifar", "CIFAR10Dataset"],
            "cifar100": ["datasets.cifar", "CIFAR100Dataset"],}

def get_methods(args):
    model = build_model(args.backbone_name, args.layers_to_extract_from, args.agg_type, args.bpm, args.patchsize, args.thres,
                args.anomaly_scorer_num_nn, args.faiss_on_gpu, args.faiss_num_workers, 
                args.topk, args.lmda, args.temp, args.return_topk_index)
    sampler = build_sampler(args.sampler, args.core_p)
    dataset = build_dataset(args.dataset, args.data_path, args.subdatasets, args.train_val_split, 
                                args.batch_size, args.num_workers,
                                args.resize, args.imagesize, args.augment, args.low_shot, args.use_multiclass)
    return [model, sampler, dataset]


def main(args):
    
    if args.use_sweep:
        wandb.init(config=vars(args))
        args = wandb.config
        
    methods = {key: item for (key, item) in get_methods(args)}

    run_save_path = uniformaly.utils.create_storage_folder(
        args.results_path, args.log_project, args.log_group, mode="iterate"
    )

    list_of_dataloaders = methods["get_dataloaders"](args.seed)

    device = uniformaly.utils.set_torch_device(args.gpu)
    # Device context here is specifically set and used later
    # because there was GPU memory-bleeding which I could only fix with
    # context managers.
    device_context = (
        torch.cuda.device("cuda:{}".format(device.index))
        if "cuda" in device.type.lower()
        else contextlib.suppress()
    )

    result_collect = []
    clustering_result_collect = []

    for dataloader_count, dataloaders in enumerate(list_of_dataloaders):
        LOGGER.info(
            "Evaluating dataset [{}] ({}/{})...".format(
                dataloaders["training"].name,
                dataloader_count + 1,
                len(list_of_dataloaders),
            )
        )

        # fix seed
        uniformaly.utils.fix_seeds(args.seed)

        dataset_name = dataloaders["training"].name

        with device_context:
            torch.cuda.empty_cache()
            imagesize = (3, args.imagesize, args.imagesize)
            sampler = methods["get_sampler"](
                device,
            )
            
            uniformaly_list = methods["get_uniformaly"](imagesize, sampler, device)

            for i, sf in enumerate(uniformaly_list):
                torch.cuda.empty_cache()
                LOGGER.info(
                    "Training models ({}/{})".format(i + 1, len(uniformaly_list))
                )
                torch.cuda.empty_cache()
                
                # half-precision
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    sf.fit(dataloaders["training"])

            torch.cuda.empty_cache()
            
            aggregator = {"scores": [], "segmentations": []}

            for i, sf in enumerate(uniformaly_list):
                torch.cuda.empty_cache()
                LOGGER.info(
                    "Embedding test data with models ({}/{})".format(
                        i + 1, len(uniformaly_list)
                    )
                )

                if args.return_topk_index:
                    scores, segmentations, labels_gt, masks_gt, topk_features = sf.predict(
                        dataloaders["testing"]
                    )
                else:
                    scores, segmentations, labels_gt, masks_gt = sf.predict(
                        dataloaders["testing"]
                    )

                aggregator["scores"].append(scores)
                aggregator["segmentations"].append(segmentations)

            scores = np.array(aggregator["scores"])
            min_scores = scores.min(axis=-1).reshape(-1, 1)
            max_scores = scores.max(axis=-1).reshape(-1, 1)
            scores = (scores - min_scores) / (max_scores - min_scores)
            scores = np.mean(scores, axis=0)

            segmentations = np.array(aggregator["segmentations"])
            min_scores = (
                segmentations.reshape(len(segmentations), -1)
                .min(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            max_scores = (
                segmentations.reshape(len(segmentations), -1)
                .max(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            segmentations = (segmentations - min_scores) / (max_scores - min_scores)
            segmentations = np.mean(segmentations, axis=0)

            # (Optional) Plot example images.
            if args.save_segmentation_images:
                image_paths = [
                    x[2] for x in dataloaders["testing"].dataset.data_to_iterate
                ]
                mask_paths = [
                    x[3] for x in dataloaders["testing"].dataset.data_to_iterate
                ]

                def image_transform(image):
                    in_std = np.array(
                        dataloaders["testing"].dataset.transform_std
                    ).reshape(-1, 1, 1)
                    in_mean = np.array(
                        dataloaders["testing"].dataset.transform_mean
                    ).reshape(-1, 1, 1)
                    image = dataloaders["testing"].dataset.transform_img(image)
                    return np.clip(
                        (image.numpy() * in_std + in_mean) * 255, 0, 255
                    ).astype(np.uint8)

                def mask_transform(mask):
                    return dataloaders["testing"].dataset.transform_mask(mask).numpy()

                image_save_path = os.path.join(
                    run_save_path, "segmentation_images", dataset_name
                )
                os.makedirs(image_save_path, exist_ok=True)
                uniformaly.utils.plot_segmentation_images(
                    image_save_path,
                    image_paths,
                    segmentations,
                    scores,
                    mask_paths,
                    image_transform=image_transform,
                    mask_transform=mask_transform,
                )

            # Clustering
            if args.return_topk_index:
                LOGGER.info("Clustering Anomalies.")
                
                cluster = uniformaly.cluster.KMeansCluster(
                                    feature=topk_features, 
                                    agg_mode='mean', 
                                    classname=dataset_name)
                _, cluster_pred = cluster.get_cluster()
                _cluster_label = [x[1] for x in dataloaders['testing'].dataset.data_to_iterate]
                label_map = {v:k for k,v in enumerate(set(_cluster_label))}
                cluster_label = [label_map[c] for c in _cluster_label]

                cluster_eval = cluster.evaluate(cluster_pred, cluster_label)
                cluster.visualize()
                clustering_result_collect.append(
                    {
                        "dataset_name": dataset_name,
                        "normalized_mutual_info": cluster_eval['nmi'],
                        "adjusted_random_idx": cluster_eval['ari'],
                        "f1_score": cluster_eval['f1'],
                    }
                )

            LOGGER.info("Computing evaluation metrics.")
            auroc = uniformaly.metrics.compute_imagewise_retrieval_metrics(
                scores, labels_gt
            )["auroc"]

            if "mvtec" in dataset_name:
                # Compute PW Auroc for all images
                pixel_scores = uniformaly.metrics.compute_pixelwise_retrieval_metrics(
                    segmentations, masks_gt
                )
                full_pixel_auroc = pixel_scores["auroc"]

                # Compute PW Auroc only images with anomalies
                sel_idxs = []
                for i in range(len(masks_gt)):
                    if np.sum(masks_gt[i]) > 0:
                        sel_idxs.append(i)
                pixel_scores = uniformaly.metrics.compute_pixelwise_retrieval_metrics(
                    [segmentations[i] for i in sel_idxs],
                    [masks_gt[i] for i in sel_idxs],
                )
                anomaly_pixel_auroc = pixel_scores["auroc"]

            else:
                full_pixel_auroc = -1.0
                anomaly_pixel_auroc = -1.0

            result_collect.append(
                {
                    "dataset_name": dataset_name,
                    "instance_auroc": auroc,
                    "full_pixel_auroc": full_pixel_auroc,
                    "anomaly_pixel_auroc": anomaly_pixel_auroc,
                }
            )

            for key, item in result_collect[-1].items():
                if key != "dataset_name":
                    LOGGER.info("{0}: {1:4.4f}".format(key, item))

            if args.return_topk_index:
                for key, item in clustering_result_collect[-1].items():
                    if key != "dataset_name":
                        LOGGER.info("{0}: {1:4.4f}".format(key, item))


            if args.save_model:
                uniformaly_save_path = os.path.join(
                    run_save_path, "models", dataset_name
                )
                os.makedirs(uniformaly_save_path, exist_ok=True)
                for i, sf in enumerate(uniformaly_list):
                    prepend = (
                        "Ensemble-{}-{}_".format(i + 1, len(uniformaly_list))
                        if len(uniformaly_list) > 1
                        else ""
                    )
                    sf.save_to_path(uniformaly_save_path, prepend)

        LOGGER.info("\n\n-----\n")

    # Store all results and mean scores to a csv-file.
    result_metric_names = list(result_collect[-1].keys())[1:]
    result_dataset_names = [results["dataset_name"] for results in result_collect]
    result_scores = [list(results.values())[1:] for results in result_collect]
    uniformaly.utils.compute_and_store_final_results(
        run_save_path,
        result_scores,
        column_names=result_metric_names,
        row_names=result_dataset_names,
    )

    if args.return_topk_index:
        clustering_result_metric_names = list(clustering_result_collect[-1].keys())[1:]
        clustering_result_dataset_names = [results["dataset_name"] for results in clustering_result_collect]
        clustering_result_scores = [list(results.values())[1:] for results in clustering_result_collect]
        uniformaly.utils.compute_and_store_final_results(
            run_save_path,
            clustering_result_scores,
            column_names=clustering_result_metric_names,
            row_names=clustering_result_dataset_names,
        )

    if args.use_wandb:
        column_names = ["dataset"] + result_metric_names
        rows = []
        
        for dn, scores in zip(result_dataset_names, result_scores):
            rows.append([dn] + scores)

        mean_metrics = {}
        for i, result_key in enumerate(column_names[1:]):
            mean_metrics[result_key] = np.mean([x[i] for x in result_scores])

        mean_metric_values = list(mean_metrics.values())
        wandb.log({
            "Mean AUROC (image)": mean_metric_values[0],
            "Mean AUROC (pixel)": mean_metric_values[1],
            "Mean AUROC (anomaly_pixel)": mean_metric_values[2]
        })

        rows.append(["Mean"] + mean_metric_values)
        
        table = wandb.Table(columns = column_names, data=rows)

        wandb.log({"metrics":table})

        if args.return_topk_index:
            column_names = ["dataset"] + clustering_result_metric_names
            rows = []
            
            for dn, scores in zip(clustering_result_dataset_names, clustering_result_scores):
                rows.append([dn] + scores)

            mean_metrics = {}
            for i, result_key in enumerate(column_names[1:]):
                mean_metrics[result_key] = np.mean([x[i] for x in clustering_result_scores])

            mean_metric_values = list(mean_metrics.values())
            wandb.log({
                "NMI": mean_metric_values[0],
                "ARI": mean_metric_values[1],
                "F1": mean_metric_values[2]
            })

            rows.append(["Mean"] + mean_metric_values)
            
            c_table = wandb.Table(columns = column_names, data=rows)

            wandb.log({"clustering_metrics":c_table})


def build_model(
    backbone_name,
    layers_to_extract_from,
    agg_type,
    bpm,
    patchsize,
    thres,
    anomaly_scorer_num_nn,
    faiss_on_gpu,
    faiss_num_workers,
    topk,
    lmda,
    temp=0.0,
    return_topk_index=False,
    **kwargs
):

    params = locals()

    def get_uniformaly(input_shape, sampler, device):
        loaded_uniformalies = []
        backbone = uniformaly.encoder.load(backbone_name)
        backbone.name = backbone_name

        local_nn_method = uniformaly.common.FaissNN(faiss_on_gpu, faiss_num_workers, device, prenorm=bool(temp))

        uniformaly_instance = uniformaly.uniformaly.Uniformaly(device, params)
        uniformaly_instance.load(
            backbone=backbone,
            layers_to_extract_from=layers_to_extract_from,
            agg_type=agg_type,
            device=device,
            input_shape=input_shape,
            bpm=bpm,
            patchsize=patchsize,
            featuresampler=sampler,
            anomaly_scorer_num_nn=anomaly_scorer_num_nn,
            local_nn_method=local_nn_method,
            lmda=lmda,
            topk=topk,
            thres=thres,
            return_topk_index=return_topk_index,
            )
        loaded_uniformalies.append(uniformaly_instance)
        return loaded_uniformalies

    return ("get_uniformaly", get_uniformaly)


def build_sampler(name, percentage):
    def get_sampler(device):
        if name == "identity":
            return uniformaly.sampler.IdentitySampler()
        elif name == "greedy_coreset":
            return uniformaly.sampler.GreedyCoresetSampler(percentage, device)
        elif name == "approx_greedy_coreset":
            return uniformaly.sampler.ApproximateGreedyCoresetSampler(percentage, device)
        else:
            return uniformaly.sampler.RandomSampler(percentage)

    return ("get_sampler", get_sampler)


def build_dataset(
    name,
    data_path,
    subdatasets,
    train_val_split,
    batch_size,
    num_workers,
    resize,
    imagesize,
    augment,
    low_shot,
    use_multiclass
):
    dataset_info = _DATASETS[name]
    dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1]])
    
    if use_multiclass:
        subdatasets = [subdatasets]
    
    def get_dataloaders(seed):
        dataloaders = []
        for subdataset in subdatasets:
            train_dataset = dataset_library.__dict__[dataset_info[1]](
                data_path,
                classname=subdataset,
                resize=resize,
                train_val_split=train_val_split,
                imagesize=imagesize,
                split="train",
                seed=seed,
                augment=augment,
                low_shot=low_shot
            )

            test_dataset = dataset_library.__dict__[dataset_info[1]](
                data_path,
                classname=subdataset,
                resize=resize,
                imagesize=imagesize,
                split="test",
                seed=seed,
            )
            
            # for semantic anomaly detection
            if name in ["cifar10", "cifar100", "species", "imagenet"]:
                train_dataset = train_dataset.dataset
                test_dataset = test_dataset.dataset
                
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

            train_dataloader.name = name
            if subdataset is not None:
                train_dataloader.name += "_"  + str(subdataset).replace("'", "").replace(", ", "")

            if train_val_split < 1:
                val_dataset = dataset_library.__dict__[dataset_info[1]](
                    data_path,
                    classname=subdataset,
                    resize=resize,
                    train_val_split=train_val_split,
                    imagesize=imagesize,
                    split="val",
                    seed=seed,
                )

                val_dataloader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True,
                )
            else:
                val_dataloader = None
            dataloader_dict = {
                "training": train_dataloader,
                "validation": val_dataloader,
                "testing": test_dataloader,
            }

            dataloaders.append(dataloader_dict)
        return dataloaders

    return ("get_dataloaders", get_dataloaders)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    
    # model
    model_arg = parser.add_argument_group("model_arg")
    model_arg.add_argument("--backbone_name", "-b", type=str, default="dino_vitb8")
    model_arg.add_argument("--layers_to_extract_from", "-le", nargs='+', type=str)
    model_arg.add_argument("--agg_type", type=str, default='layer', choices=['layer', 'channel', 'cat', 'patch'])
    model_arg.add_argument("--bpm", action="store_true")
    model_arg.add_argument("--patchsize", type=int, default=7)
    model_arg.add_argument("--thres", type=float, default=0.1)
    model_arg.add_argument("--anomaly_scorer_num_nn", type=int, default=1)
    model_arg.add_argument("--lmda", type=float, default=0.0)
    model_arg.add_argument("--temp", type=float, default=0.0)
    model_arg.add_argument("--topk", type=float, default=0.01)
    model_arg.add_argument("--return_topk_index", action="store_true")
    
    
    # sampler
    sampler_arg = parser.add_argument_group("sampler_arg")
    sampler_arg.add_argument("--sampler", type=str, default="approx_greedy_coreset")
    sampler_arg.add_argument("-p", "--core_p", type=float, default=0.01)

    # dataset
    dataset_arg = parser.add_argument_group("dataset_arg")
    dataset_arg.add_argument("dataset", type=str, default="mvtec")
    dataset_arg.add_argument("data_path", type=str, default="/home/data/anomaly_detection")
    dataset_arg.add_argument("--subdatasets", "-d", type=str, nargs='+')
    dataset_arg.add_argument("--train_val_split", type=float, default=1.0)
    dataset_arg.add_argument("--resize", type=int, default=256)
    dataset_arg.add_argument("--imagesize", type=int, default=224)
    dataset_arg.add_argument("--augment", action="store_true")
    dataset_arg.add_argument("--use_multiclass", action="store_true")
    dataset_arg.add_argument("--low_shot", type=int, default=-1)

    # misc
    misc_arg = parser.add_argument_group("misc_arg")
    misc_arg.add_argument("--seed", type=int, default=0)
    misc_arg.add_argument("--gpu", type=int, nargs='+', default=0)
    misc_arg.add_argument("--num_workers", type=int, default=4)
    misc_arg.add_argument("--faiss_on_gpu", action="store_true")
    misc_arg.add_argument("--faiss_num_workers", type=int, default=8)
    misc_arg.add_argument("--batch_size", type=int, default=1)

    # logging
    logging_arg = parser.add_argument_group("logging_arg")
    logging_arg.add_argument("--results_path", type=str, default="/home/data/exp/uniformaly_results")
    logging_arg.add_argument("--log_group", type=str, default="group")
    logging_arg.add_argument("--log_project", type=str, default="project")
    logging_arg.add_argument("--save_segmentation_images", action="store_true")
    logging_arg.add_argument("--save_model", action="store_true")

    # wandb
    wandb_arg = parser.add_argument_group("wandb_arg")
    wandb_arg.add_argument("--use_wandb", action="store_true")
    wandb_arg.add_argument("--wandb_project", type=str, default="uniformaly")
    wandb_arg.add_argument("--wandb_run", type=str, default="")
    wandb_arg.add_argument("--use_sweep", action="store_true")


    args = parser.parse_args()
    print(f"[Arguments] \n ====> {vars(args)}")

    if args.use_wandb:
        if args.use_sweep:
            
            from functools import partial
            
            sweep_config = {
            'name' : 'seed-test', # sweep name
            'method': 'grid', # grid, bayes
            'metric' : {
                'name': 'Mean AUROC (image)',
                'goal': 'maximize'
                },
            'parameters' : {
                'seed': {
                    'max': 1000,
                    'min': 0
                    },
                'batch_size': {
                        'values': [1,2,4,8,16,32,64,128,256,512]
                    },
                'anomaly_scorer_nn':{
                        'max': 10,
                        'min': 1
                    }
                },
            }
            
            sweep_id = wandb.sweep(sweep_config, project=args.wandb_project ,entity="kfai")
            main = partial(main, args)
            wandb.agent(sweep_id, main, count=100)
            torch.cuda.empty_cache()
            
        else:
            wandb.init(project=args.wandb_project, name=f"{args.wandb_run}_{args.dataset}_{args.backbone_name}_top{args.topk}_{args.sampler}{args.core_p}", entity="kfai", config=vars(args))
            main(args)
            
        wandb.finish()

    else:
        main(args)