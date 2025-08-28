import argparse
from pathlib import Path

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
#from lightning.pytorch.callbacks import ModelSummary
from pytorch_lightning.loggers import WandbLogger
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
    resume_checkpoint: Optional[str] = None


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


def main(args):
    # for repeatability
    seed_everything(args.seed, workers=True)

    print("Starting a new run with the following parameters:")
    print(args)

    params_str = f"{'noval ' if not args.val else ''}{'hung ' if args.hung_data else ''}{'fold' + str(args.fold) + ' ' if args.fold is not None else ''}{args.loss}-h{args.transformer_dim}-aug{args.tempo_augmentation}{args.pitch_augmentation}{args.mask_augmentation}{' nosumH ' if not args.sum_head else ''}{' nopartialT ' if not args.partial_transformers else ''}"
    if args.logger == "wandb":
        if args.resume_checkpoint and args.resume_id:
            wandb_args = dict(id=args.resume_id, resume="must")
        else:
            wandb_args = {}
        logger = WandbLogger(
            project="beat_this", name=f"{args.name} {params_str}".strip(), **wandb_args
        )
    else:
        logger = None

    if args.force_flash_attention:
        print("Forcing the use of the flash attention.")
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(False)

    data_dir =Path(args.data_path) / "data" #Path(__file__).parent.parent.relative_to(Path.cwd()) / "data"
    print(data_dir)
    checkpoint_dir = Path(args.checkpoint_path) 
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
        partial_transformers=args.partial_transformers,
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
    # save only the last model
    callbacks.append(
        ModelCheckpoint(
            every_n_epochs=1,
            dirpath=str(checkpoint_dir),
            filename=f"{args.name} S{args.seed} {params_str}".strip(),
        )
        
    )
    use_gpu = torch.cuda.is_available() 
    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if use_gpu else "cpu",
        devices=1 if use_gpu else 1, 
        num_sanity_val_steps=1,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=1,
        precision="16-mixed",
        accumulate_grad_batches=args.accumulate_grad_batches,
        check_val_every_n_epoch=args.val_frequency,
    )
    current_state = pl_model.state_dict()
    #print(list(current_state.keys())) # this model doesn't have orig suffix
    ckpt = torch.load(args.resume_checkpoint, map_location="cpu")

# In Lightning checkpoints the model weights are under "state_dict"
    ckpt_state = ckpt["state_dict"]

    print("== Current model keys ==")
    print(list(pl_model.state_dict().keys())[:20])

    print("\n== Checkpoint keys ==")
    print(list(ckpt_state.keys())[:20])
    trainer.fit(pl_model, datamodule, ckpt_path=args.resume_checkpoint)
    trainer.test(pl_model, datamodule)


if __name__ == "__main__":
    cfg =load_config("launch_scripts/train_params.yaml")
   # args = parser.parse_args()

    main(cfg)
