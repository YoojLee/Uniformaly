import argparse
import contextlib
import gc
import logging
import os
import sys

import numpy as np
import torch

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
            "cifar": ["datasets.cifar", "CIFAR10Dataset"],
            "ood": ["datasets.ood", "OODDataset"],}


def get_methods(args):
    model = load_model(args.model_path, args.topk, args.thres, args.patchsize, args.anomaly_scorer_nn, args.faiss_on_gpu, args.faiss_num_workers, args.gpu)
    
    if not args.ood:
        dataset = build_dataset(args.dataset, args.data_path, args.subdatasets,
                                args.batch_size, args.num_workers,
                                args.resize, args.imagesize, args.use_multiclass)
    else:
        dataset = build_ood_dataset(args.dataset, args.data_path, args.in_dataset, args.subdatasets,
                                    args.batch_size, args.num_workers, args.resize,
                                    args.imagesize)
    
    return [model, dataset]



def main(args):
    methods = {key: item for (key, item) in get_methods(args)}

    os.makedirs(args.results_path, exist_ok=True)

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

    dataloader_iter, n_dataloaders = methods["get_dataloaders_iter"]
    dataloader_iter = dataloader_iter(args.seed)
    uniformaly_iter, n_uniformalys = methods["get_uniformaly_iter"]
    uniformaly_iter = uniformaly_iter(device)
    
    if not (n_dataloaders == n_uniformalys or n_uniformalys == 1):
        raise ValueError(
            "Please ensure that #uniformalys == #Datasets or #uniformalys == 1!"
        )

    for dataloader_count, dataloaders in enumerate(dataloader_iter):
        LOGGER.info(
            "Evaluating dataset [{}] ({}/{})...".format(
                dataloaders["testing"].name, dataloader_count + 1, n_dataloaders
            )
        )

        uniformaly.utils.fix_seeds(args.seed, device)

        dataset_name = args.dataset

        with device_context:

            torch.cuda.empty_cache()
            if dataloader_count < n_uniformalys:
                uniformaly_list = next(uniformaly_iter)

            aggregator = {"scores": [], "segmentations": []}
            for i, sf in enumerate(uniformaly_list):
                torch.cuda.empty_cache()
                LOGGER.info(
                    "Embedding test data with models ({}/{})".format(
                        i + 1, len(uniformaly_list)
                    )
                )
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

            
            # Plot Example Images.
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

                uniformaly.utils.plot_segmentation_images(
                    args.results_path,
                    image_paths,
                    segmentations,
                    scores,
                    mask_paths,
                    image_transform=image_transform,
                    mask_transform=mask_transform,
                )

            LOGGER.info("Computing evaluation metrics.")
            # Compute Image-level AUROC scores for all images.
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
                    "dataset_name": dataloaders["testing"].name,
                    "instance_auroc": auroc,
                    "full_pixel_auroc": full_pixel_auroc,
                    "anomaly_pixel_auroc": anomaly_pixel_auroc,
                }
            )

            for key, item in result_collect[-1].items():
                if key != "dataset_name":
                    LOGGER.info("{0}: {1:4.4f}".format(key, item))

            gc.collect()

        LOGGER.info("\n\n-----\n")

    result_metric_names = list(result_collect[-1].keys())[1:]
    result_dataset_names = [results["dataset_name"] for results in result_collect]
    result_scores = [list(results.values())[1:] for results in result_collect]
    uniformaly.utils.compute_and_store_final_results(
        args.results_path,
        result_scores,
        column_names=result_metric_names,
        row_names=result_dataset_names,
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
            "Mean AUROC (image)": mean_metric_values[0]
        })

        rows.append(["Mean"] + mean_metric_values)
        
        table = wandb.Table(columns = column_names, data=rows)

        wandb.log({"metrics":table})


def load_model(model_path, topk, thres, patchsize, anomaly_scorer_nn, faiss_on_gpu, faiss_num_workers, gpu):
    def get_uniformaly_iter(device):
        for path in model_path:
            loaded_uniformalys = []
            gc.collect()
            n_uniformalys = len(
                [x for x in os.listdir(path) if ".faiss" in x]
            )
            if n_uniformalys == 1:
                local_nn_method = uniformaly.common.FaissNN(faiss_on_gpu, faiss_num_workers, device, prenorm=False)

                uniformaly_instance = uniformaly.uniformaly.Uniformaly(device, None)
                uniformaly_instance.load_from_path(
                    load_path=path, device=device, 
                    local_nn_method=local_nn_method,
                    anomaly_scorer_nn=anomaly_scorer_nn,
                    topk=topk, thres=thres, patchsize=patchsize
                )
                loaded_uniformalys.append(uniformaly_instance)
            else:
                for i in range(n_uniformalys):
                    local_nn_method = uniformaly.common.FaissNN(faiss_on_gpu, faiss_num_workers, device, prenorm=False)
                    uniformaly_instance = uniformaly.uniformaly.uniformaly(device, None)
                    uniformaly_instance.load_from_path(
                        load_path=path,
                        device=device,
                        local_nn_method=local_nn_method,
                        anomaly_scorer_nn=anomaly_scorer_nn,
                        prepend="Ensemble-{}-{}_".format(i + 1, n_uniformalys),
                    )
                    loaded_uniformalys.append(uniformaly_instance)

            yield loaded_uniformalys
    return ("get_uniformaly_iter", [get_uniformaly_iter, len(model_path)])



def build_dataset(
    name, data_path, subdatasets, batch_size, num_workers, resize, imagesize, use_multiclass
):
    dataset_info = _DATASETS[name]
    dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1]])

    if use_multiclass:
        subdatasets = [subdatasets]

    def get_dataloaders_iter(seed):
        for subdataset in subdatasets:
            test_dataset = dataset_library.__dict__[dataset_info[1]](
                data_path,
                classname=subdataset,
                resize=resize,
                imagesize=imagesize,
                split="test",
                seed=seed,
            )
            
            if name in ["cifar", "species"]:
                test_dataset = test_dataset.dataset

            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

            test_dataloader.name = name
            if subdataset is not None:
                test_dataloader.name += "_" + str(subdataset).replace("'", "").replace(", ", "")

            dataloader_dict = {"testing": test_dataloader}

            yield dataloader_dict

    return ("get_dataloaders_iter", [get_dataloaders_iter, len(subdatasets)])


def build_ood_dataset(
    name, data_path, in_dataset, out_datasets, batch_size, num_workers, resize, imagesize
):
    dataset_info = _DATASETS[name]
    dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1]])

    def get_dataloaders_iter(seed):
        
        test_dataset_in = dataset_library.__dict__[dataset_info[1]](
                data_path,
                dataset=in_dataset,
                resize=resize,
                imagesize=imagesize,
                split="test",
                seed=seed,
                is_ood=False
            )
        
        for out_dataset in out_datasets:
            test_dataset_out = dataset_library.__dict__[dataset_info[1]](
                data_path,
                dataset=out_dataset,
                resize=resize,
                imagesize=imagesize,
                split="test",
                seed=seed,
                is_ood=True
            )

            test_dataset = test_dataset_in + test_dataset_out
            
            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

            test_dataloader.name = name
            if out_dataset is not None:
                test_dataloader.name += "_" + str(out_dataset)

            dataloader_dict = {"testing": test_dataloader}

            yield dataloader_dict

    return ("get_dataloaders_iter", [get_dataloaders_iter, len(out_datasets)])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    
    # model
    model_arg = parser.add_argument_group("model_arg")
    model_arg.add_argument("--model_path", type=str, nargs='+')
    model_arg.add_argument("--topk", type=float, default=0.05)
    model_arg.add_argument("--thres", type=float, default=0.1)
    model_arg.add_argument("--patchsize", type=int, default=7)
    model_arg.add_argument("--anomaly_scorer_nn", type=int, default=1)

    # dataset
    dataset_arg = parser.add_argument_group("dataset_arg")
    dataset_arg.add_argument("dataset", type=str, default="mvtec")
    dataset_arg.add_argument("data_path", type=str, default="/home/data/anomaly_detection")
    dataset_arg.add_argument("--subdatasets", "-d", type=str, nargs='+')
    dataset_arg.add_argument("--resize", type=int, default=256)
    dataset_arg.add_argument("--imagesize", type=int, default=224)
    dataset_arg.add_argument("--augment", action="store_true")
    dataset_arg.add_argument("--use_multiclass", action="store_true")
    dataset_arg.add_argument("--ood", action="store_true")
    dataset_arg.add_argument("--in_dataset", type=str, default="")

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
    logging_arg.add_argument("--results_path", type=str, default="results")
    logging_arg.add_argument("--save_segmentation_images", action="store_true")

    # wandb
    wandb_arg = parser.add_argument_group("wandb_arg")
    wandb_arg.add_argument("--use_wandb", action="store_true")
    wandb_arg.add_argument("--wandb_project", type=str, default="uniformaly")
    wandb_arg.add_argument("--wandb_run", type=str, default="exp1")

    args = parser.parse_args()
    print(f"[Arguments] \n ====> {vars(args)}")

    if args.use_wandb:
        wandb.init(project=args.wandb_project, name=f"{args.wandb_run}_{args.dataset}_evaluation", entity="kfai", config=vars(args))

    main(args)

    if args.use_wandb:
        wandb.finish()
