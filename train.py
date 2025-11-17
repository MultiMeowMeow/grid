from __future__ import annotations

from pathlib import Path
from typing import Optional

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
import pytorch_lightning as pl
import torch
from torch_geometric.datasets import OPFDataset
from torch_geometric.loader import DataLoader

from core import OPFCore
from flows import constraint_losses
from opf_normalization import OPFNormalizer


class OPFDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self._root: Optional[str] = None
        self.train_set: Optional[OPFDataset] = None
        self.val_set: Optional[OPFDataset] = None
        self.test_set: Optional[OPFDataset] = None

    def _resolve_root(self) -> str:
        if self._root is None:
            self._root = to_absolute_path(self.cfg.root)
        return self._root

    def _build_dataset(self, split: str) -> OPFDataset:
        root = self._resolve_root()
        return OPFDataset(
            root=root,
            split=split,
            case_name=self.cfg.case_name,
            num_groups=self.cfg.num_groups,
            topological_perturbations=self.cfg.topological_perturbations,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in ("fit", None):
            if self.train_set is None:
                self.train_set = self._build_dataset("train")
            if self.val_set is None:
                self.val_set = self._build_dataset("val")
        if stage in ("test", None):
            if self.test_set is None:
                self.test_set = self._build_dataset("test")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_set,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            persistent_workers=self.cfg.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_set,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            persistent_workers=self.cfg.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_set,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            persistent_workers=self.cfg.num_workers > 0,
        )


class OPFLitModule(pl.LightningModule):
    def __init__(self, model_cfg: DictConfig, optim_cfg: DictConfig, loss_cfg: DictConfig, normalizer_path: Path):
        super().__init__()
        self.save_hyperparameters(ignore=["normalizer_path"])
        self.model = OPFCore(
            num_layers=model_cfg.num_layers,
            hidden_size=model_cfg.hidden_size
        )
        self.model.norm = OPFNormalizer.load(str(normalizer_path))
        self.loss_cfg = loss_cfg
        self.optim_cfg = optim_cfg

    def forward(self, batch):  # type: ignore[override]
        return self.model(batch)

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        bus_pred, gen_pred = self(batch)
        losses = constraint_losses(
            data=batch,
            va=bus_pred[:, 0],
            vm=bus_pred[:, 1],
            pg=gen_pred[:, 0],
            qg=gen_pred[:, 1],
            w_eq=self.loss_cfg.w_eq,
            w_th=self.loss_cfg.w_th,
            w_ang=self.loss_cfg.w_ang,
        )
        self.log(f"{stage}/loss", losses["total"], prog_bar=True, batch_size=batch["bus"].x.size(0))
        self.log(f"{stage}/eq", losses["eq"], batch_size=batch["bus"].x.size(0))
        self.log(f"{stage}/thermal", losses["thermal"], batch_size=batch["bus"].x.size(0))
        if self.loss_cfg.w_ang != 0.0:
            self.log(f"{stage}/angle", losses["angle"], batch_size=batch["bus"].x.size(0))
        return losses["total"]

    def training_step(self, batch, batch_idx):  # type: ignore[override]
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):  # type: ignore[override]
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):  # type: ignore[override]
        self._shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.optim_cfg.lr,
            weight_decay=self.optim_cfg.weight_decay,
        )
        return optimizer


def prepare_normalizer(cfg: DictConfig) -> Path:
    stats_path = Path(to_absolute_path(cfg.normalizer.stats_path))
    if stats_path.exists() and not cfg.normalizer.overwrite:
        return stats_path

    root = to_absolute_path(cfg.data.root)
    train_dataset = OPFDataset(
        root=root,
        split="train",
        case_name=cfg.data.case_name,
        num_groups=cfg.data.num_groups,
        topological_perturbations=cfg.data.topological_perturbations,
    )
    normalizer = OPFNormalizer().fit(train_dataset)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    normalizer.save(str(stats_path))
    return stats_path


@hydra.main(config_path="configs", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    pl.seed_everything(42, workers=True)
    normalizer_path = prepare_normalizer(cfg)

    data_module = OPFDataModule(cfg.data)
    module = OPFLitModule(cfg.model, cfg.optimizer, cfg.loss, normalizer_path)

    trainer = pl.Trainer(**cfg.trainer)
    trainer.fit(module, datamodule=data_module)


if __name__ == "__main__":
    main()
