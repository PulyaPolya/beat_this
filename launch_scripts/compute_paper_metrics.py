import argparse
from pathlib import Path
import json
import numpy as np
from pytorch_lightning import Trainer, seed_everything
import cProfile
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from beat_this.dataset import BeatDataModule
from beat_this.inference import load_checkpoint
from beat_this.model.pl_module import PLBeatThis
from collections import OrderedDict
import json
import torch
from pathlib import Path
from copy import deepcopy


def load_checkpoint_resume(seed_folder, seed, epoch= None):
    seed_folder = seed_folder.replace("_SEED_", str(seed))
    checkpoint_type = "best" if epoch == "best" else "periodic"
    #print(checkpoint_type)
    checkpoint_folder = os.path.join(seed_folder, checkpoint_type)
    if checkpoint_type == "best":
        checkpoint = [check for check in os.listdir(checkpoint_folder) if "orig" not in check ][0]
    else:
        epoch = epoch -1
        checkpoint = [check for check in os.listdir(checkpoint_folder) if f"epoch={epoch}" in check and "orig" not in check ][0]
    checkpoint_path = os.path.join(checkpoint_folder, checkpoint)
    print(f"Loading {checkpoint_type} checkpoint for seed {seed} from folder {seed_folder}  from the path {checkpoint_path}")
    #checkpoint_name = Path(checkpoint_path).stem
    #kpt = torch.load(checkpoint_path, map_location="cpu")
    #print(ckpt)
    
    return checkpoint_path
# for repeatability
seed_everything(0, workers=True)


def main(args):
    if len(args.models) == 1:
        # single model prediction
        checkpoint_path = load_checkpoint_resume(seed_folder = args.models[0], seed = args.seed, epoch = args.epoch)
        checkpoint = load_checkpoint(checkpoint_path)
        print("Single model prediction for", checkpoint_path)
        if not args.debug:
            use_gpu = 0 if torch.cuda.is_available() and args.gpu >= 0 else -1
            print(f"Using GPU: {use_gpu}")
            # create datamodule
            datamodule = datamodule_setup(checkpoint, args)
            # create model and trainer
            model, trainer = plmodel_setup(
                checkpoint, args.eval_trim_beats, args.dbn, use_gpu
            )
            # predict
            metrics, dataset, preds, piece, dict_all_results = compute_predictions(
                model, trainer, datamodule.predict_dataloader()
            )
       # save predictions to a json file
        out_file_name =  Path(args.models[0]).stem
        subfolder = args.subfolder if args.subfolder is not None else ""
        if args.name:
            name = args.name
        else:
            name = ""
        if args.clustering_config is not None and args.cluster_number is not None:
            save_path = os.path.join(f"json_{args.datasplit}_scores", args.clustering_config, f"cluster_{args.cluster_number}", name)
        else:
            dir = "all_data_metrics" if args.all_metrics ==True else "full_data"
            cluster= str(args.cluster_number) if args.cluster_number is not None else ""
            save_path = os.path.join(f"json_{args.datasplit}_scores", dir, subfolder, cluster, name)

        # elif args.all_metrics ==True:
        #     save_path = os.path.join(f"json_{args.datasplit}_scores", "all_data_metrics", name)
        # else:
        #     save_path = os.path.join(f"json_{args.datasplit}_scores", "full_data", name)
        os.makedirs(save_path, exist_ok = True)
        test_scores_path = os.path.join(save_path, f"epoch_{args.epoch}_seed_{args.seed}_{out_file_name}.json")
        print(test_scores_path)
        if not args.debug:
            with open(test_scores_path, 'w') as fp:
                json.dump(dict_all_results, fp)
            averaged_metrics = {k: np.mean(v) for k, v in metrics.items()}
            # compute metrics averaged by dataset
            dataset_metrics = {
                k: {d: np.mean(v[dataset == d]) for d in np.unique(dataset)}
                for k, v in metrics.items()
            }
            # print for dataset
            print("Metrics")
            for k, v in averaged_metrics.items():
                print(f"{k}: {v}")
            print("Dataset metrics")
            for k, v in dataset_metrics.items():
                print(k)
                for d, value in v.items():
                    print(f"{d}: {value}")
                print("------")
    else:  # multiple models
        if args.aggregation_type == "mean-std":
            # computing result variability for the same dataset and different model seeds
            # create datamodule only once, as we assume it is the same for all models
            checkpoint = load_checkpoint(args.models[0])
            datamodule = datamodule_setup(checkpoint, args.num_workers, args.datasplit)
            # create model and trainer
            all_metrics = []
            for checkpoint_path in args.models:
                checkpoint = load_checkpoint(checkpoint_path)
                model, trainer = plmodel_setup(
                    checkpoint, args.eval_trim_beats, args.dbn, use_gpu
                )

                metrics, dataset, preds, piece, dict_all_results = compute_predictions(
                    model, trainer, datamodule.predict_dataloader()
                )
                # compute averaged metrics for one model
                averaged_metrics = {k: np.mean(v) for k, v in metrics.items()}
                all_metrics.append(averaged_metrics)
            # compute mean and standard deviations for all model averages
            all_metrics_mean = {
                k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]
            }
            all_metrics_std = {
                k: np.std([m[k] for m in all_metrics]) for k in all_metrics[0]
            }
            all_metrics_stats = {
                k: (all_metrics_mean[k], all_metrics_std[k])
                for k, v in all_metrics[0].items()
            }
            # print all metrics
            print("Metrics")
            for k, v in all_metrics_stats.items():
                # round to 3 decimal places
                print(f"{k}: {round(v[0],3)} +- {round(v[1],3)}")
        elif args.aggregation_type == "k-fold":
            # computing results in the K-fold setting. Every fold has a different dataset
            all_piece_metrics = []
            all_piece_dataset = []
            all_piece = []
            # create datamodule for each model
            for i_model, checkpoint_path in enumerate(args.models):
                print(f"Model {i_model+1}/{len(args.models)}")
                checkpoint = load_checkpoint(checkpoint_path)
                datamodule = datamodule_setup(
                    checkpoint, args.num_workers, args.datasplit
                )
                # create model and trainer
                model, trainer = plmodel_setup(
                    checkpoint, args.eval_trim_beats, args.dbn, args.gpu
                )
                # predict
                metrics, dataset, preds, piece = compute_predictions(
                    model, trainer, datamodule.predict_dataloader()
                )
                all_piece_metrics.append(metrics)
                all_piece_dataset.append(dataset)
                all_piece.append(piece)
            # aggregate across folds
            all_piece_metrics = {
                k: np.concatenate([m[k] for m in all_piece_metrics])
                for k in all_piece_metrics[0]
            }
            all_piece_dataset = np.concatenate(all_piece_dataset)
            all_piece = np.concatenate(all_piece)
            # double check that there are no errors in the fold and there are not repeated pieces
            assert len(all_piece) == len(
                np.unique(all_piece)
            ), "There are repeated pieces in the folds"
            dataset_metrics = {
                k: {
                    d: np.mean(v[all_piece_dataset == d])
                    for d in np.unique(all_piece_dataset)
                }
                for k, v in all_piece_metrics.items()
            }
            # print for dataset
            print("Dataset metrics")
            for k, v in dataset_metrics.items():
                print(k)
                for d, value in v.items():
                    print(f"{d}: {round(value,3)}")
                print("------")
        else:
            raise ValueError(f"Unknown aggregation type {args.aggregation_type}")


def datamodule_setup(checkpoint, args):
    # Load the datamodule
    print("Creating datamodule")
    if args.clustering_config is not None:
        data_path = Path(args.data_path) / args.clustering_config / f"cluster_{args.cluster_number}"
    else:
        data_path = Path(args.data_path)
    data_dir = data_path / "data" #Path(__file__).parent.parent.relative_to(Path.cwd()) / "data"
    datamodule_hparams = checkpoint["datamodule_hyper_parameters"]
    # update the hparams with the ones from the arguments
    if args.num_workers is not None:
        datamodule_hparams["num_workers"] = args.num_workers
    datamodule_hparams["predict_datasplit"] = args.datasplit
    datamodule_hparams["data_dir"] = data_dir
    datamodule_hparams["files"] = None
    #print(datamodule_hparams)
    datamodule = BeatDataModule(**datamodule_hparams)
    datamodule.setup(stage="predict")
    return datamodule


def plmodel_setup(checkpoint, eval_trim_beats, dbn, gpu):
    """
    Set up the pytorch lightning model and trainer for evaluation.

    Args:
        checkpoint_path (dict): The dict containing the checkpoint to load.
        eval_trim_beats (int or None): The number of beats to trim during evaluation. If None, the setting is taken from the pretrained model.
        dbn (bool or None): Whether to use the Dynamic Bayesian Network (DBN) module during evaluation. If None, the default behavior from the pretrained model is used.
        gpu (int): The index of the GPU device to use for training.

    Returns:
        tuple: A tuple containing the initialized pytorch lightning model and trainer.

    """
    if eval_trim_beats is not None:
        checkpoint["hyper_parameters"]["eval_trim_beats"] = eval_trim_beats
    if dbn is not None:
        checkpoint["hyper_parameters"]["use_dbn"] = dbn

    model = PLBeatThis(**checkpoint["hyper_parameters"])
    model.load_state_dict(checkpoint["state_dict"])
    # set correct device and accelerator
    if gpu >= 0:
        devices = [gpu]
        accelerator = "gpu"
    else:
        devices = 1
        accelerator = "cpu"
    # create trainer
    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        logger=False,
        deterministic=True,
        precision="16-mixed",
        inference_mode=True
    )
    return model, trainer


def compute_predictions(model, trainer, predict_dataloader):
    print("Computing predictions ...")
    out = trainer.predict(model, predict_dataloader)
    metrics = [o[0] for o in out]
    dict_all_results = {out[i][3][0]: out[i][0] for i in range(len(out))}
    preds = [o[1] for o in out]
    dataset = np.asarray([o[2][0] for o in out])
    piece = np.asarray([o[3][0] for o in out])
    # convert metrics from list of per-batch dictionaries to a single dictionary with np arrays as values
    metrics = {k: np.asarray([m[k] for m in metrics]) for k in metrics[0]}
    return metrics, dataset, preds, piece, dict_all_results 

def load_args_from_json(json_file):
    """Load arguments from a JSON file."""
    with open(json_file, "r") as f:
        data = json.load(f)
        #return json.load(f)
    class Args:
        def __init__(self, **entries):
            self.__dict__.update(entries)

    return Args(**data)

if __name__ == "__main__":
    
    args =  load_args_from_json("helpers/evaluate_params.json")
    import cProfile
    import pstats

    # profiler = cProfile.Profile()
    # profiler.enable()
    args0 = deepcopy(args)

    for seed in args0.seed_list:
        args.seed = seed
        main(args)
    # Run function
 

    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats("cumulative")
    # stats.print_stats(50)
