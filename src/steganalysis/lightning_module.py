from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from lightning import LightningDataModule, LightningModule
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig, LRSchedulerConfigType
from sklearn.utils import shuffle
import torch
from torchvision.transforms.v2 import Compose, RandomApply, RandomChoice, RandomHorizontalFlip, RandomRotation, RandomVerticalFlip, ToImage
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, F1Score, Precision, Recall, Specificity, MatthewsCorrCoef

from steganalysis.dataset import SteganalysisDataset
from steganalysis.model import InvResNet


@dataclass
class DatasetParameter:
	steganalysis_method: Literal["WOW", "S_UNIWARD"]
	payload: Literal["1,0bpp", "0,8bpp", "0,6bpp", "0,4bpp", "0,2bpp"]
	downscaling_method: Literal["Nearest", "Bicubic"]
	batch_size: int


class SteganalysisDataModule(LightningDataModule):
	def __init__(self, dataset_path: str | Path, params: DatasetParameter) -> None:
		super().__init__()
		self.dataset_path = Path(dataset_path)
		self.params = params
		self.transform = Compose([
			RandomApply([
				RandomChoice([
					RandomRotation([90, 90]),
					RandomRotation([180, 180]),
					RandomRotation([270, 270])
				])
			], 0.75),
			RandomVerticalFlip(),
			RandomHorizontalFlip()
		])
		return
	
	def setup(self, stage) -> None:
		shuffled_idx = shuffle(range(10000), random_state=42)
		if shuffled_idx is not None:
			self.shuffled = [int(i) for i in shuffled_idx]
		bossbase_cover = self.dataset_path / f"BossBase-cover-{self.params.downscaling_method}.tar"
		bossbase_stego = self.dataset_path / f"BossBase-{self.params.steganalysis_method}-{self.params.payload}-{self.params.downscaling_method}.tar"
		bows2_cover = self.dataset_path / f"BOWS2-cover-{self.params.downscaling_method}.tar"
		bows2_stego = self.dataset_path / f"BOWS2-{self.params.steganalysis_method}-{self.params.payload}-{self.params.downscaling_method}.tar"

		bossbase_with_transform = SteganalysisDataset(bossbase_cover, bossbase_stego, self.transform)
		bows2_with_transform = SteganalysisDataset(bows2_cover, bows2_stego, self.transform)
		bossbase = SteganalysisDataset(bossbase_cover, bossbase_stego)

		if stage == "fit":
			if shuffled_idx is not None:
				self.ds_train = torch.utils.data.ConcatDataset([
					torch.utils.data.Subset(bossbase_with_transform, self.shuffled[:4000]),
					bows2_with_transform
				])
				self.ds_valid = torch.utils.data.Subset(bossbase, self.shuffled[4000:5000])
		
		if stage == "test":
			self.ds_test = torch.utils.data.Subset(bossbase, self.shuffled[5000:])
		
		return
	
	def train_dataloader(self) -> torch.utils.data.DataLoader:
		return torch.utils.data.DataLoader(self.ds_train, self.params.batch_size, True)
	
	def val_dataloader(self) -> torch.utils.data.DataLoader:
		return torch.utils.data.DataLoader(self.ds_valid, self.params.batch_size, False)
	
	def test_dataloader(self) -> torch.utils.data.DataLoader:
		return torch.utils.data.DataLoader(self.ds_test, self.params.batch_size, False)


class LitInvResNet(LightningModule):
	def __init__(self) -> None:
		super().__init__()
		self.model = InvResNet()
		self.sigmoid = torch.nn.Sigmoid()
		self.criterion = torch.nn.BCEWithLogitsLoss()

		metrics = MetricCollection({
			"accuracy": Accuracy("binary"),
			"f1": F1Score("binary"),
			"precision": Precision("binary"),
			"recall": Recall("binary"), # also known as sensitivity
			"specificity": Specificity("binary"),
			"mcc": MatthewsCorrCoef("binary")
		})

		self.train_metrics = metrics.clone("train_")
		self.val_metrics = metrics.clone("val_")
		self.test_metrics = metrics.clone("test_")
		return
	
	def forward(self, images) -> torch.Tensor:
		return self.model(images)
	
	def training_step(self, batch, batch_idx) -> torch.Tensor:
		covers, cover_labels, stegos, stego_labels = batch
		images = torch.cat([covers, stegos])
		labels = torch.cat([cover_labels, stego_labels])
		preds = self(images)

		self.train_metrics.update(self.sigmoid(preds), labels)

		loss = self.criterion(preds, labels)
		self.log("train_loss", loss, on_step=False, on_epoch=True)
		return loss
	
	def on_train_epoch_end(self) -> None:
		self.log_dict(self.train_metrics.compute())
		self.train_metrics.reset()
		return
	
	def validation_step(self, batch, batch_idx) -> None:
		covers, cover_labels, stegos, stego_labels = batch
		images = torch.cat([covers, stegos])
		labels = torch.cat([cover_labels, stego_labels])
		preds = self(images)

		self.val_metrics.update(self.sigmoid(preds), labels)

		loss = self.criterion(preds, labels)
		self.log("val_loss", loss, on_step=False, on_epoch=True)
		return
	
	def on_validation_epoch_end(self) -> None:
		self.log_dict(self.val_metrics.compute())
		self.val_metrics.reset()
		return
	
	def test_step(self, batch, batch_idx) -> None:
		covers, cover_labels, stegos, stego_labels = batch
		images = torch.cat([covers, stegos])
		labels = torch.cat([cover_labels, stego_labels])
		preds = self(images)

		self.test_metrics.update(self.sigmoid(preds), labels)

		loss = self.criterion(preds, labels)
		self.log("test_loss", loss, on_step=False, on_epoch=True)
		return
	
	def on_test_epoch_end(self) -> None:
		self.log_dict(self.test_metrics.compute())
		self.test_metrics.reset()
		return
	
	def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
		optimizer = torch.optim.SGD(self.parameters(), 0.0001, 0.9, weight_decay=0.00001)
		steps = int(self.trainer.estimated_stepping_batches)
		lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 0.01, steps)
		lr_scheduler_config: LRSchedulerConfigType = {
			"scheduler": lr_scheduler,
			"interval": "step"
		}
		return {
			"optimizer": optimizer,
			"lr_scheduler": lr_scheduler_config
		}
