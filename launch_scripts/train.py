import argparse
from pathlib import Path
import math
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, Callback, EarlyStopping
#from lightning.pytorch.callbacks import ModelSummary
from pytorch_lightning.loggers import WandbLogger
from collections import OrderedDict
from torchinfo import summary
import sys
import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional
from pathlib import Path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from beat_this.dataset import BeatDataModule
from beat_this.model.pl_module import PLBeatThis
import yaml
import torch.nn as nn
import random
import json
from pathlib import Path
import numpy as np
from beat_this.inference import load_checkpoint

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # # For full reproducibility
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def freeze_by_prefix(module: nn.Module, prefixes):
    if isinstance(prefixes, str):
        prefixes = [prefixes]
    for name, p in module.named_parameters():
        if any(name.startswith(pref) for pref in prefixes):
            p.requires_grad = False

def unfreeze_by_prefix(module: nn.Module, prefixes):
    unfrozen_params = []
    if isinstance(prefixes, str):
        prefixes = [prefixes]
    for name, p in module.named_parameters():
        if any(name.startswith(pref) for pref in prefixes):
            p.requires_grad = True
            unfrozen_params.append(p)
    return unfrozen_params


def set_bn_eval_by_prefix(module: nn.Module, prefixes):
    if isinstance(prefixes, str):
        prefixes = [prefixes]
    for n, m in module.named_modules():
        if any(n.startswith(pref) for pref in prefixes):
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.eval()

def freeze_layers(num_to_freeze, pl_model):
    layers_to_freeze = [ "model.frontend._orig_mod"]
    for i in range(num_to_freeze):
        layers_to_freeze.append(f"model.transformer_blocks._orig_mod.layers.{i}")
    freeze_by_prefix(pl_model, layers_to_freeze)

    set_bn_eval_by_prefix(pl_model, layers_to_freeze)

    #return pl_model
class PlateauUnfreeze(Callback):
    def __init__(self, monitor="val_f1", mode="max", patience=1,  lr_backbone=1e-5):
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        #self.get_blocks_fn = get_blocks_fn
        self.lr_backbone = lr_backbone
        self.best = -float("inf") if mode=="max" else float("inf")
        self.bad_epochs = 0
        self._num_unfrozen = 0
        self._layers = None

    def on_fit_start(self, trainer, pl_module):
        # Discover your exact stack once:
        # model.transformer_blocks._orig_mod.layers is a ModuleList
        self._layers = list(pl_module.model.transformer_blocks._orig_mod.layers)
        self._report_trainable(pl_module, prefix=" (start)")
    
    def _report_trainable(self, pl_module, prefix=""):
        total = sum(p.numel() for p in pl_module.parameters())
        trainable = sum(p.numel() for p in pl_module.parameters() if p.requires_grad)
        print(f"[PlateauUnfreeze]{prefix} Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    def on_validation_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        if self.monitor not in metrics:
            return
        current = metrics[self.monitor].item()

        improved = (current > self.best) if self.mode=="max" else (current < self.best)
        if improved:
            self.best = current
            self.bad_epochs = 0
            return

        self.bad_epochs += 1
        if self.bad_epochs < self.patience:
            return
        #print("unfreezing new layers")
        # Unfreeze next block
        unfrozen = sum(any(p.requires_grad for p in L.parameters()) for L in self._layers)
        next_idx = len(self._layers) - 1 - unfrozen
        if next_idx < 0:
            return  # nothing left to unfreeze
        
        prefix = f"model.transformer_blocks._orig_mod.layers.{next_idx}."
        new_params = unfreeze_by_prefix(pl_module, prefix)
        # if new_params:
        #     trainer.optimizers[0].add_param_group({"params": new_params, "lr": self.lr_backbone})
        self._num_unfrozen += 1
        self.bad_epochs = 0
        self._report_trainable(pl_module, prefix=f" (after unfreezing layer {next_idx})")
        # trainer.logger.log_metrics({"unfrozen_blocks": self._num_unfrozen}, step=trainer.global_step)

        # lr =trainer.optimizers[0].param_groups[0]["lr"]
        # trainer.logger.log_metrics({"lr": lr}, step=trainer.global_step)

@dataclass
class Config:
    name: str = ""
    gpu: int = 0
    force_flash_attention: bool = False
    compile: List[str] = field(default_factory=lambda: ["frontend", "transformer_blocks", "task_heads"])
    n_layers: int = 6
    transformer_dim: int = 512
    frontend_dropout: float = 0.1
    transformer_dropout: float = 0.2
    lr: float = 8e-4
    weight_decay: float = 0.01
    logger: str = "none"  # or "wandb"
    num_workers: int = 8
    n_heads: int = 16
    fps: int = 50
    loss: str = "shift_tolerant_weighted_bce"  # one of: shift_tolerant_weighted_bce, fast_shift_tolerant_weighted_bce, weighted_bce, bce
    warmup_steps: int = 1000
    max_epochs: int = 100
    batch_size: int = 8
    accumulate_grad_batches: int = 8
    train_length: int = 1500
    dbn: bool = False
    eval_trim_beats: float = 5.0
    val_frequency: int = 5
    tempo_augmentation: bool = True
    pitch_augmentation: bool = True
    mask_augmentation: bool = True
    sum_head: bool = True
    partial_transformers: bool = True
    length_based_oversampling_factor: float = 0.65
    val: bool = True
    hung_data: bool = False
    fold: Optional[int] = None
    seed: int = 0
    data_path : str = "data"
    checkpoint_path : str = "data"
    resume_checkpoint: bool = False
    resume_id :  Optional[str] = None
    # when none take the best checkpoint for this seed, else take the periodic checkpoint at this epoch
    ckpt_epoch : Optional[int] = None  # 
    ## only used for continuing training
    checkpoints_folder: Optional[str] = None
    freeze_layers: Optional[int] = None
    save_frequency: Optional[int] = None
    use_early_stopping: bool = True                              
    es_patience: int = 10                    
    es_min_delta: float = 0.001   
    compute_metrics: bool =  False
    #full_data : bool = False
    cluster_number : Optional[int]  = None
    clustering_config: Optional[str] = None


def _load_yaml_or_json(path: Path) -> dict:
    raw = path.read_text()
    if path.suffix.lower() in (".yml", ".yaml"):
        data = yaml.safe_load(raw) or {}
    return data

def load_config(path: str | os.PathLike) -> Config:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    data = _load_yaml_or_json(p)
    # allow hyphenated keys in file
    data = {k.replace("-", "_"): v for k, v in data.items()}
    # validate keys
    valid = set(Config.__annotations__.keys())
    unknown = set(data) - valid
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    return Config(**data)


def compute_predictions(model, trainer, predict_dataloader, checkpoint_name):
    print("Computing predictions ...")#
    out = trainer.predict(model, predict_dataloader, ckpt_path=checkpoint_name)
    metrics = [o[0] for o in out]
    dict_all_results = {out[i][3][0]: out[i][0] for i in range(len(out))}
    preds = [o[1] for o in out]
    dataset = np.asarray([o[2][0] for o in out])
    piece = np.asarray([o[3][0] for o in out])
    # convert metrics from list of per-batch dictionaries to a single dictionary with np arrays as values
    metrics = {k: np.asarray([m[k] for m in metrics]) for k in metrics[0]}
    return metrics, dataset, preds, piece, dict_all_results

def compute_metrics(model, trainer, checkpoint_name, data_dir, save_predictions = False, cluster_info = None, datasplit = "test", num_workers=16, name = None):
    checkpoint = load_checkpoint(checkpoint_name)
    datamodule_hparams =  checkpoint["datamodule_hyper_parameters"]
    # update the hparams with the ones from the arguments

    datamodule_hparams["num_workers"] = num_workers
    datamodule_hparams["predict_datasplit"] = datasplit
    datamodule_hparams["data_dir"] = data_dir
    datamodule = BeatDataModule(**datamodule_hparams)
    datamodule.setup(stage="predict")
    metrics, dataset, preds, piece, dict_all_results = compute_predictions(
            model, trainer, datamodule.predict_dataloader(), checkpoint_name)
       # save predictions to a json file
    if save_predictions:
        out_file_name =  Path(checkpoint_name).stem
        if cluster_info:
            save_path = os.path.join(f"json_{datasplit}_scores", cluster_info[0],cluster_info[1])
            # os.makedirs(save_path, exist_ok = True)
            # test_scores_path = os.path.join(save_path, f"{out_file_name}.json")
        else:
            subfolder = name if name else "" 
            save_path = os.path.join(f"json_{datasplit}_scores", subfolder)
        os.makedirs(save_path, exist_ok=True)
        test_scores_path = os.path.join(save_path,  f"{out_file_name}.json")

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

def load_checkpoint_resume(seed_folder, seed, epoch= None):
    seed_folder = seed_folder.replace("_SEED_", str(seed))
    checkpoint_type = "best" if epoch == None else "periodic"
    checkpoint_folder = os.path.join(seed_folder, checkpoint_type)
    if checkpoint_type == "best":
        checkpoint = [check for check in os.listdir(checkpoint_folder) if "orig" in check][0]
    else:
        epoch = epoch -1
        checkpoint = [check for check in os.listdir(checkpoint_folder) if f"{epoch:02d}" in check][0]
    checkpoint_path = os.path.join(checkpoint_folder, checkpoint)
    print(f"Loading {checkpoint_type} checkpoint for seed {seed} from folder {seed_folder}  from the path {checkpoint_path}")
    checkpoint_name = Path(checkpoint_path).stem
    if not checkpoint_name.endswith("_orig"):
        ckpt = rename_best_checkpoint(checkpoint_path,  save = False)
    else:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
    return ckpt

def rename_key(key: str, insert: str) -> str:
    parts = key.split(".")
    if len(parts) > 2:
        parts.insert(2, insert)   # insert after the 2nd dot
    return ".".join(parts) 

def rename_best_checkpoint(best_ckpt_path,  key_fragment="_orig_mod", save = True):
    ckpt = torch.load(best_ckpt_path, map_location="cpu")
    sd = ckpt.get("state_dict", None)
    if sd is None:
        raise KeyError(f"'state_dict' not in checkpoint: {best_ckpt_path}")
    old_sd = ckpt["state_dict"]
    new_sd = OrderedDict((rename_key(k, "_orig_mod"), v) for k, v in old_sd.items())
    ckpt["state_dict"] = new_sd
    new_name = Path(best_ckpt_path).stem + "_orig.ckpt"
    folder = os.path.dirname(best_ckpt_path)
    new_path = os.path.join(folder, new_name )
    if save:
        torch.save(ckpt,new_path)  #
        return new_path
    else:
        return ckpt

def main(args):
    # for repeatability
    seed_everything(args.seed, workers=True)
    set_seed(args.seed)

    print("Starting  new run with the following parameters:")
    print(args)

    params_str = f"{'noval ' if not args.val else ''}{'hung ' if args.hung_data else ''}{'fold' + str(args.fold) + ' ' if args.fold is not None else ''}{args.loss}-h{args.transformer_dim}-aug{args.tempo_augmentation}{args.pitch_augmentation}{args.mask_augmentation}{' nosumH ' if not args.sum_head else ''}{' nopartialT ' if not args.partial_transformers else ''}"
    if args.logger == "wandb":
        if args.resume_checkpoint:
            #wandb_args = dict(id=args.resume_id, resume="must")
            group = args.clustering_config
        else:
            #wandb_args = {}
            group = None
        logger = WandbLogger(
            project="beat_this", name=args.name, group = group, config = vars(args)
        )
    else:
        logger = None

    if args.force_flash_attention:
        print("Forcing the use of the flash attention.")
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(False)
    if args.cluster_number:
        print(f"Using data from cluster number {args.cluster_number}")
        data_dir =Path(os.path.join(args.data_path, args.clustering_config, f"cluster_{args.cluster_number}")) / "data" #Path(__file__).parent.parent.relative_to(Path.cwd()) / "data"
    else:
        data_dir =Path(args.data_path) / "data" 
    print(data_dir)
    
    #checkpoint_dir = Path(args.checkpoint_path) 
    #(Path(__file__).parent.parent.relative_to(Path.cwd()) / "checkpoints")
    augmentations = {}
    if args.tempo_augmentation:
        augmentations["tempo"] = {"min": -20, "max": 20, "stride": 4}
    if args.pitch_augmentation:
        augmentations["pitch"] = {"min": -5, "max": 6}
    if args.mask_augmentation:
        # kind, min_count, max_count, min_len, max_len, min_parts, max_parts
        augmentations["mask"] = {
            "kind": "permute",
            "min_count": 1,
            "max_count": 6,
            "min_len": 0.1,
            "max_len": 2,
            "min_parts": 5,
            "max_parts": 9,
        }

    datamodule = BeatDataModule(
        data_dir,
        batch_size=args.batch_size,
        train_length=args.train_length,
        spect_fps=args.fps,
        num_workers=args.num_workers,
        test_dataset="gtzan",
        length_based_oversampling_factor=args.length_based_oversampling_factor,
        augmentations=augmentations,
        hung_data=args.hung_data,
        no_val=not args.val,
        fold=args.fold,
    )
    datamodule.setup(stage="fit")

    # compute positive weights
    pos_weights = datamodule.get_train_positive_weights(widen_target_mask=3)
    print("Using positive weights: ", pos_weights)
    dropout = {
        "frontend": args.frontend_dropout,
        "transformer": args.transformer_dropout,
    }
    pl_model = PLBeatThis(
        spect_dim=128,
        fps=50,
        transformer_dim=args.transformer_dim,
        ff_mult=4,
        n_layers=args.n_layers,
        stem_dim=32,
        dropout=dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        pos_weights=pos_weights,
        head_dim=32,
        loss_type=args.loss,
        warmup_steps=args.warmup_steps,
        max_epochs=args.max_epochs,
        use_dbn=args.dbn,
        eval_trim_beats=args.eval_trim_beats,
        sum_head=args.sum_head,
        partial_transformers=args.partial_transformers
    )
    #print(ModelSummary(model=pl_model, max_depth=2)) 
    print(summary(pl_model))
    for part in args.compile:
        if hasattr(pl_model.model, part):
            setattr(pl_model.model, part, torch.compile(getattr(pl_model.model, part)))
            print("Will compile model", part)
        else:
            raise ValueError("The model is missing the part", part, "to compile")

    callbacks = [LearningRateMonitor(logging_interval="step")]
    # save every 5 epochs
    if not args.cluster_number: #args.full_data:
        checkpoint_folder = os.path.join(args.checkpoint_path, f"{args.name}S{args.seed}{params_str}".strip() )
    else:
        #data_path_parts = args.data_path.split(os.sep)
        cluster_info = [args.clustering_config, str(args.cluster_number)]
        checkpoint_folder = os.path.join(args.checkpoint_path,cluster_info[0], cluster_info[1], f"{args.name}S{args.seed}{params_str}".strip() )
    if args.save_frequency:
        save_top_k = -1
        every_n_epochs=args.save_frequency
    else:
        save_top_k = 1
        every_n_epochs=args.val_frequency
    if args.save_frequency:
        periodic_ckpt_cb =(
        ModelCheckpoint(
                dirpath=os.path.join(checkpoint_folder, "periodic"),
                filename="{epoch:02d}-valf{val_F-measure_beat:.4f}",
                save_top_k=save_top_k,
                monitor=None,
                every_n_epochs=every_n_epochs,
            )
         )
        callbacks.append(periodic_ckpt_cb)
            
   
    best_ckpt_cb = ModelCheckpoint(
        dirpath=os.path.join(checkpoint_folder, "best"),
        filename=f"best-seed{args.seed}-{{epoch:02d}}-valf{{val_F-measure_beat:.4f}}",
        monitor="val_F-measure_beat",
        mode="max",
        save_top_k=1,                   # keep only the best
        save_last=False,                # optional; can set True if desired
    )
    callbacks.append(best_ckpt_cb)
    if args.use_early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor="val_F-measure_beat",    
                mode="max",       
                patience=max(1, int(math.ceil(args.es_patience / args.val_frequency))),  
                min_delta=args.es_min_delta, 
                verbose=True
            )
        )
    # callbacks.append(
    # PlateauUnfreeze(
    #     monitor="val_F-measure_beat",      # choose your metric name
    #     mode="max",            # because higher F1 is better
    #     patience=3,            # wait 3 bad epochs before unfreezing
    #     #get_blocks_fn=get_blocks_fn,
    #     lr_backbone=5e-5
    # )
    # )
    use_gpu = torch.cuda.is_available() 
    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if use_gpu else "cpu",
        devices=1 if use_gpu else 1, 
        num_sanity_val_steps=0,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=1,
        precision="16-mixed",
        accumulate_grad_batches=args.accumulate_grad_batches,
        check_val_every_n_epoch=args.val_frequency,
    )
    if args.resume_checkpoint:

        #ckpt = torch.load(args.resume_checkpoint, map_location="cpu")
        
        param_name = "model.frontend.stem.bn1d.weight" 
        before = pl_model.state_dict()[param_name].clone()
        # Load weights
        ckpt = load_checkpoint_resume(seed_folder=args.checkpoints_folder, seed=args.seed, epoch=args.ckpt_epoch)
        missing, unexpected = pl_model.load_state_dict(ckpt["state_dict"], strict=False)
        print("Loaded weights. Missing:", missing)
        print("Unexpected:", unexpected)
        after = pl_model.state_dict()[param_name]
        print("Are weights identical?", torch.equal(before, after))

        if args.freeze_layers:
            freeze_layers(args.freeze_layers, pl_model)
            total = sum(p.numel() for p in pl_model.parameters())
            trainable = sum(p.numel() for p in pl_model.parameters() if p.requires_grad)
            print(f"Trainable: {trainable:,} / {total:,}")
    
        print(f"validating the model before")    # uncomment  for real runs!!!!
        trainer.validate(pl_model, datamodule=datamodule )
    trainer.fit(pl_model, datamodule)
    best_path = best_ckpt_cb.best_model_path 
    print("Best checkpoint:", best_path)
    print(f"validating the model after")
    # rename some weights for error-free processing
    new_best_path = rename_best_checkpoint(best_path)
    trainer.validate(pl_model, datamodule=datamodule, ckpt_path=new_best_path)
    cluster_info = None if not args.cluster_number else cluster_info
    compute_metrics(pl_model, trainer, checkpoint_name = new_best_path, data_dir = data_dir, save_predictions = True, cluster_info = cluster_info, datasplit = "val",
                     num_workers=args.num_workers, name=args.name)
    #compute_metrics(pl_model, trainer, datamodule, checkpoint_name = new_best_path, save_predictions= True, cluster_info= cluster_info, setup="val")
    #trainer.test(pl_model, datamodule, ckpt_path =new_best_path)
    if args.compute_metrics:
        compute_metrics(pl_model, trainer, datamodule, checkpoint_name = new_best_path, save_predictions= True, cluster_info= cluster_info, setup="test",
                        name = args.name)
        

    #trainer.test(pl_model, datamodule)

if __name__ == "__main__":
    cfg =load_config("launch_scripts/train_params.yaml")
   # args = parser.parse_args()

    main(cfg)
