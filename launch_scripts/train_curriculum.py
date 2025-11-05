import argparse
from pathlib import Path

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, Callback
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
import torch.nn as nn
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For full reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




    #return pl_model
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, ConcatDataset

class ConcatBeatDataModule(LightningDataModule):
    def __init__(self, dms, batch_size, num_workers, shuffle=True):
        super().__init__()
        self.dms = dms
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self._train = None
        self._val = None
        self._test = None

    def setup(self, stage=None):
        # Ensure each child DM is set up so its loaders are ready
        for dm in self.dms:
            dm.setup(stage="fit")

        # Collect underlying datasets from loaders
        train_sets = []
        val_sets = []
        test_sets = []

        for dm in self.dms:
            tr = dm.train_dataloader()
            if tr is not None:
                train_sets.append(tr.dataset)
            vl = dm.val_dataloader() if hasattr(dm, "val_dataloader") else None
            if vl is not None:
                val_sets.append(vl.dataset)
            # ts = dm.test_dataloader() if hasattr(dm, "test_dataloader") else None
            # if ts is not None:
            #     test_sets.append(ts.dataset)

        if train_sets:
            self._train = ConcatDataset(train_sets)
        if val_sets:
            self._val = ConcatDataset(val_sets)
        if test_sets:
            self._test = ConcatDataset(test_sets)

    def train_dataloader(self):
        return DataLoader(
            self._train, batch_size=self.batch_size,
            shuffle=self.shuffle, num_workers=self.num_workers, pin_memory=True
        )

    def val_dataloader(self):
        if self._val is None:
            return None
        return DataLoader(
            self._val, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers, pin_memory=True
        )

    def test_dataloader(self):
        if self._test is None:
            return None
        return DataLoader(
            self._test, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers, pin_memory=True
        )


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
    resume_id :  Optional[str] = None
    freeze_layers: Optional[int] = None
    save_frequency: Optional[int] = None
    curriculum_dirs: Optional[List[str]] = None   # e.g., ["cluster_0","cluster_1","cluster_2","cluster_3"]
    curriculum_stage_epochs: Optional[List[int]] = None  # e.g., [10, 10, 10, 10]
    run_quick_test: bool = False
    #wandb_name : Optional[str] = None

def get_val_len(dm) -> int:
    # make sure the validation dataset exists
    dm.setup("validate")
    val_loader = dm.val_dataloader()
    if val_loader is None:
        return 0
    return len(val_loader.dataset)

def build_datamodule_for_root(root_dirs, args: Config):
    # Single root → behave exactly as before
    if isinstance(root_dirs, (str, Path)):
        root_dirs = Path(root_dirs)
        dm = BeatDataModule(
            root_dirs,
            batch_size=args.batch_size,
            train_length=args.train_length,
            spect_fps=args.fps,
            num_workers=args.num_workers,
            test_dataset="gtzan",
            length_based_oversampling_factor=args.length_based_oversampling_factor,
            augmentations={
                **({"tempo": {"min": -20, "max": 20, "stride": 4}} if args.tempo_augmentation else {}),
                **({"pitch": {"min": -5, "max": 6}} if args.pitch_augmentation else {}),
                **({
                    "mask": {"kind": "permute", "min_count": 1, "max_count": 6,
                             "min_len": 0.1, "max_len": 2, "min_parts": 5, "max_parts": 9}
                } if args.mask_augmentation else {}),
            },
            hung_data=args.hung_data,
            no_val=not args.val,
            fold=args.fold,
        )
        dm.setup(stage="fit")
        return dm

    # List of roots → cumulative curriculum
    dm_list = []
    for rd in root_dirs:
        dm = BeatDataModule(
            rd,
            batch_size=args.batch_size,
            train_length=args.train_length,
            spect_fps=args.fps,
            num_workers=args.num_workers,
            test_dataset="gtzan",
            length_based_oversampling_factor=args.length_based_oversampling_factor,
            augmentations={
                **({"tempo": {"min": -20, "max": 20, "stride": 4}} if args.tempo_augmentation else {}),
                **({"pitch": {"min": -5, "max": 6}} if args.pitch_augmentation else {}),
                **({
                    "mask": {"kind": "permute", "min_count": 1, "max_count": 6,
                             "min_len": 0.1, "max_len": 2, "min_parts": 5, "max_parts": 9}
                } if args.mask_augmentation else {}),
            },
            hung_data=args.hung_data,
            no_val=not args.val,
            fold=args.fold,
        )
        dm_list.append(dm)

    concat_dm = ConcatBeatDataModule(
        dms=dm_list,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )
    concat_dm.setup(stage="fit")
    return concat_dm


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

class LogCumulativeEpoch(Callback):
    """Re-log metrics to W&B with a continuous epoch counter across stages."""
    def __init__(self, offset: int = 0):
        super().__init__()
        self.offset = offset
        self._defined = False

    def _epoch_cum(self, trainer):
        # 1-based display; change to +0 if you prefer 0-based
        return int(self.offset + trainer.current_epoch + 1)

    def _define_metrics_once(self, trainer):
        if self._defined:
            return
        if isinstance(trainer.logger, WandbLogger):
            wb = trainer.logger.experiment
            # Tell W&B that all charts should use epoch_cum as x-axis
            wb.define_metric("epoch_cum")
            wb.define_metric("*", step_metric="epoch_cum")
            self._defined = True

    def _log_with_epoch_cum(self, trainer):
        if not isinstance(trainer.logger, WandbLogger):
            return
        self._define_metrics_once(trainer)
        epoch_cum = self._epoch_cum(trainer)

        # Collect latest epoch-level metrics that Lightning aggregates for us
        metrics = {}
        for k, v in trainer.callback_metrics.items():
            try:
                metrics[k] = float(v.detach().cpu().item())  # tensors → float
            except Exception:
                try:
                    metrics[k] = float(v)
                except Exception:
                    continue

        # Also send epoch_cum itself; set step=epoch_cum so W&B aligns points
        trainer.logger.log_metrics({**metrics, "epoch_cum": epoch_cum}, step=epoch_cum)

    # Re-log at the end of train and validation epochs
    def on_train_epoch_end(self, trainer, pl_module):
        self._log_with_epoch_cum(trainer)

    def on_validation_epoch_end(self, trainer, pl_module):
        self._log_with_epoch_cum(trainer)

def run_curriculum(args: Config):
    # fixed logger across stages (so weights carry in one process, logs grouped)
    # if args.logger == "wandb":
    #     logger = WandbLogger(project="beat_this", name=args.name, config=vars(args))
    # else:
    #     logger = None

    # Create the model ONCE; keep weights across stages
    data_root0 = Path(args.data_path) / args.curriculum_dirs[0] / "data"
    dm0 = build_datamodule_for_root(data_root0, args)
    pos_weights0 = dm0.get_train_positive_weights(widen_target_mask=3)

    dropout = {"frontend": args.frontend_dropout, "transformer": args.transformer_dropout}
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
        pos_weights=pos_weights0,
        head_dim=32,
        loss_type=args.loss,
        warmup_steps=args.warmup_steps,
        max_epochs=args.max_epochs,   # unused here; we set per-stage in Trainer
        use_dbn=args.dbn,
        eval_trim_beats=args.eval_trim_beats,
        sum_head=args.sum_head,
        partial_transformers=args.partial_transformers
    )
    epoch_offset = 0
    print(summary(pl_model))
    for part in args.compile:
        if hasattr(pl_model.model, part):
            setattr(pl_model.model, part, torch.compile(getattr(pl_model.model, part)))
            print("Will compile model", part)
        else:
            raise ValueError("The model is missing the part", part, "to compile")

    use_gpu = torch.cuda.is_available()

    # Iterate over curriculum stages
    for stage_idx, (stage_dir, stage_epochs) in enumerate(
        zip(args.curriculum_dirs, args.curriculum_stage_epochs, strict=True)
    ):
        stage_roots =  [Path(args.data_path) / d / "data" for d in args.curriculum_dirs[:stage_idx + 1]]
        print(f"\n=== Curriculum Stage {stage_idx}: {stage_dir} ({stage_epochs} epochs) ===")
        datamodule = build_datamodule_for_root(stage_roots, args)
        logger = WandbLogger(project="beat_this", group=args.name,
                             name=f"{args.name}_stage{stage_idx}",
                             config={**vars(args), "stage": stage_idx,
                                     "epoch_offset": epoch_offset}) \
                 if args.logger == "wandb" else None

        cum_epoch_cb = LogCumulativeEpoch(offset=epoch_offset)
        val_len = get_val_len(datamodule)
        print(f"[Stage {stage_idx}] Validation set size: {val_len}")
        # If you want stage-specific positive weights, update here (safe even if unchanged)
        try:
            pos_w = datamodule.get_train_positive_weights(widen_target_mask=3)
            if hasattr(pl_model, "pos_weights"):
                pl_model.pos_weights = pos_w  # PLBeatThis uses this attribute in loss
        except Exception as e:
            print(f"Skipping pos-weight refresh for stage {stage_idx}: {e}")


        if args.save_frequency:
            save_top_k = -1
            every_n_epochs = args.save_frequency
        else:
            save_top_k = 1
            every_n_epochs = args.val_frequency

        callbacks = [
            LearningRateMonitor(logging_interval="step"),
             cum_epoch_cb
            # ModelCheckpoint(
            #     dirpath=checkpoint_folder,
            #     filename="{epoch:02d}-valf{val_F_measure_beat:.4f}",
            #     save_top_k=save_top_k,
            #     every_n_epochs=every_n_epochs,
            # ),
        ]
    
        # Fresh Trainer per stage (simplest way to bound stage epochs)
        trainer = Trainer(
            max_epochs=stage_epochs,
            accelerator="gpu" if use_gpu else "cpu",
            devices=1,
            num_sanity_val_steps=0,
            logger=logger,
            callbacks=callbacks,
            log_every_n_steps=1,
            precision="16-mixed",
            accumulate_grad_batches=args.accumulate_grad_batches,
            check_val_every_n_epoch=args.val_frequency,
            fast_dev_run=1 if args.run_quick_test else False,      # 1 train/val/test batch total
            limit_train_batches=1 if args.run_quick_test else 1.0, # can also use e.g. 0.01
            limit_val_batches=1 if args.run_quick_test else 1.0,
            limit_test_batches=1 if args.run_quick_test else 1.0,
      
        )

        # Optional: quick val before training this stage (to log baseline)
        # print(f"Validating before stage {stage_idx} training…")
        # trainer.validate(pl_model, datamodule=datamodule)

        # Train this stage
        trainer.fit(pl_model, datamodule=datamodule)

        # Optional: evaluate on fixed test (GTZAN) after each stage
        # print(f"Testing after stage {stage_idx}…")
        # trainer.test(pl_model, datamodule=datamodule)
        if logger: logger.experiment.finish()
        epoch_offset += stage_epochs

    print("\nCurriculum finished.")


def main(args):
    # for repeatability
    seed_everything(args.seed, workers=True)
    set_seed(args.seed)

    print("Starting a new run with the following parameters:")
    print(args)
    run_curriculum(args)
    
if __name__ == "__main__":
    cfg =load_config("launch_scripts/curriculum_train_params.yaml")
   # args = parser.parse_args()

    main(cfg)
