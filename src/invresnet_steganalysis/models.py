import collections.abc
import typing

import lightning
import lightning.pytorch.utilities.types
import torch
import torchmetrics
import torchvision
from torchvision.models._utils import _make_divisible


__all__ = [ "SRMConv2d", "SRMConv2dNormActivation", "InvResNet", "SpatialPyramidPool2d" ]


class SRMConv2d(torch.nn.Module):
	def __init__(self, trainable: bool = True) -> None:
		super().__init__()

		self.type1 = _srm_filter_3x3(trainable)
		self.type2 = _srm_filter_5x5(trainable)
		return

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		out1 = self.type1(x)
		out2 = self.type2(x)
		return torch.cat([out1, out2], 1)


class SRMConv2dNormActivation(torch.nn.Module):
	def __init__(self, trainable: bool = True) -> None:
		super().__init__()

		self.srm_conv = SRMConv2d(trainable)
		self.norm = torch.nn.BatchNorm2d(30)
		torch.nn.init.uniform_(self.norm.bias, -1.0, 1.0)
		self.act = torch.nn.Hardtanh(-3.0, 3.0, True)
		return

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.act(self.norm(self.srm_conv(x)))


class InvResNet(torch.nn.Module):
	def __init__(self) -> None:
		super().__init__()

		self.conv_layers = torch.nn.Sequential(
			SRMConv2dNormActivation(False),
			InvertedResidual(30, 36, 30, 3, 1, False, torch.nn.Hardtanh),
			InvertedResidual(30, 48, 30, 3, 1, False, torch.nn.SiLU),
			torch.nn.AvgPool2d(5, 2, 2),
			InvertedResidual(30, 48, 40, 3, 1, False, torch.nn.SiLU),
			InvertedResidual(40, 54, 40, 3, 1, False, torch.nn.SiLU),
			torch.nn.AvgPool2d(5, 2, 2),
			InvertedResidual(40, 54, 50, 3, 1, False, torch.nn.SiLU),
			InvertedResidual(50, 72, 50, 3, 1, True, torch.nn.SiLU),
			torch.nn.AvgPool2d(5, 5, 2),
			InvertedResidual(50, 72, 60, 3, 1, True, torch.nn.SiLU),
			InvertedResidual(60, 80, 60, 3, 1, True, torch.nn.SiLU),
		)

		self.pool_layers = torch.nn.ModuleList([
			SpatialPyramidPool2d("avg"),
			SpatialPyramidPool2d("max")
		])

		self.classifier = torch.nn.Sequential(
			torch.nn.Linear(2520, 1260),
			torch.nn.SiLU(),
			torch.nn.Dropout(p=0.2, inplace=True),
			torch.nn.Linear(1260, 1)
		)
		return

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		tmp = self.conv_layers(x)
		tmp = torch.cat([
			pool(tmp) for pool in self.pool_layers
		], 1).view(x.size(0), -1)
		return self.classifier(tmp)


class SpatialPyramidPool2d(torch.nn.Module):
	def __init__(
		self,
		mode: typing.Literal["max", "avg"],
		outputs_size: list[int | tuple[int, int]] = [1, 2, 4]
	) -> None:
		super().__init__()

		if mode == "max":
			pool_fn = torch.nn.AdaptiveMaxPool2d
		elif mode == "avg":
			pool_fn = torch.nn.AdaptiveAvgPool2d
		else:
			raise ValueError(f"Unknown pool mode {mode}, expected 'max' or 'avg'")

		self.pools = torch.nn.ModuleList([pool_fn(size) for size in outputs_size])
		return

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		batch_size = x.size(0)
		channels = x.size(1)

		out = [pool_fn(x).view(batch_size, channels, -1) for pool_fn in self.pools]
		return torch.cat(out, dim=2)


class LitInvResNet(lightning.LightningModule):
	def __init__(
		self, lr: float = 0.0001, max_lr: float = 0.01, momentum: float = 0.9, weight_decay: float = 0.00001
	) -> None:
		super().__init__()
		self.model = InvResNet()
		self.sigmoid = torch.nn.Sigmoid()
		self.criterion = torch.nn.BCEWithLogitsLoss()
		self.lr = lr
		self.max_lr = max_lr
		self.momentum = momentum
		self.weight_decay = weight_decay

		metrics = torchmetrics.MetricCollection({
			"accuracy": torchmetrics.Accuracy("binary"),
			"f1": torchmetrics.F1Score("binary"),
			"precision": torchmetrics.Precision("binary"),
			"recall": torchmetrics.Recall("binary"), # also known as sensitivity
			"specificity": torchmetrics.Specificity("binary"),
			"mcc": torchmetrics.MatthewsCorrCoef("binary")
		})

		self.train_metrics = metrics.clone("train_")
		self.val_metrics = metrics.clone("val_")
		self.test_metrics = metrics.clone("test_")
		return
	
	def forward(self, images) -> torch.Tensor:
		return self.model(images)
	
	def training_step(self, batch) -> torch.Tensor:
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
	
	def validation_step(self, batch) -> None:
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
	
	def test_step(self, batch) -> None:
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
	
	def configure_optimizers(self) -> lightning.pytorch.utilities.types.OptimizerLRSchedulerConfig:
		optimizer = torch.optim.SGD(self.model.parameters(), self.lr, self.momentum, weight_decay=self.weight_decay)
		steps = int(self.trainer.estimated_stepping_batches)
		lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, self.max_lr, steps)
		lr_scheduler_config: lightning.pytorch.utilities.types.LRSchedulerConfigType = {
			"scheduler": lr_scheduler,
			"interval": "step"
		}
		return {
			"optimizer": optimizer,
			"lr_scheduler": lr_scheduler_config
		}


class InvertedResidual(torch.nn.Module):
	def __init__(
		self,
		in_channels: int,
		expanded_channels: int,
		out_channels: int,
		kernel_size: int,
		stride: int,
		use_se: bool,
		activation_layer: collections.abc.Callable[..., torch.nn.Module],
	) -> None:
		super().__init__()

		self.use_residual = (stride == 1) and (in_channels == out_channels)

		layers: list[torch.nn.Module] = []

		# Expansion
		if expanded_channels != in_channels:
			layers.append(
				torchvision.ops.Conv2dNormActivation(
					in_channels=in_channels,
					out_channels=expanded_channels,
					kernel_size=1,
					activation_layer=activation_layer,
				)
			)

		# Depthwise + optional Squeeze-Excite
		layers.append(
			torchvision.ops.Conv2dNormActivation(
				in_channels=expanded_channels,
				out_channels=expanded_channels,
				kernel_size=kernel_size,
				stride=stride,
				groups=expanded_channels,
				activation_layer=activation_layer,
			)
		)

		if use_se:
			squeeze_channels = _make_divisible(expanded_channels // 4, 8)
			layers.append(torchvision.ops.SqueezeExcitation(expanded_channels, squeeze_channels))

		# Projection
		layers.append(
			torchvision.ops.Conv2dNormActivation(
				in_channels=expanded_channels,
				out_channels=out_channels,
				kernel_size=1,
				activation_layer=None,
			)
		)

		self.block = torch.nn.Sequential(*layers)
		return

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		res = self.block(x)
		if self.use_residual:
			res += x
		return res


class _srm_filter_3x3(torch.nn.Module):
	def __init__(self, trainable: bool = True) -> None:
		super().__init__()

		self.weight = torch.nn.Parameter(
			torch.tensor([[[
					[0.0,	0.0,	0.0],
					[0.0,	-1.0,	1.0],
					[0.0,	0.0,	0.0]
				]], [[
					[0.0,	0.0,	0.0],
					[0.0,	-1.0,	0.0],
					[0.0,	0.0,	1.0]
				]], [[
					[0.0,	0.0,	0.0],
					[0.0,	-1.0,	0.0],
					[0.0,	1.0,	0.0]
				]], [[
					[0.0,	0.0,	0.0],
					[0.0,	-1.0,	0.0],
					[1.0,	0.0,	0.0]
				]], [[
					[0.0,	0.0,	0.0],
					[1.0,	-1.0,	0.0],
					[0.0,	0.0,	0.0]
				]], [[
					[1.0,	0.0,	0.0],
					[0.0,	-1.0,	0.0],
					[0.0,	0.0,	0.0]
				]], [[
					[0.0,	1.0,	0.0],
					[0.0,	-1.0,	0.0],
					[0.0,	0.0,	0.0]
				]], [[
					[0.0,	0.0,	1.0],
					[0.0,	-1.0,	0.0],
					[0.0,	0.0,	0.0]
				]], [[
					[0.0,	0.0,	0.0],
					[1.0,	-2.0,	1.0],
					[0.0,	0.0,	0.0]
				]], [[
					[1.0,	0.0,	0.0],
					[0.0,	-2.0,	0.0],
					[0.0,	0.0,	1.0]
				]], [[
					[0.0,	1.0,	0.0],
					[0.0,	-2.0,	0.0],
					[0.0,	1.0,	0.0]
				]], [[
					[0.0,	0.0,	1.0],
					[0.0,	-2.0,	0.0],
					[1.0,	0.0,	0.0]
				]], [[
					[-1.0,	2.0,	-1.0],
					[ 2.0,	-4.0,	2.0],
					[ 0.0,	0.0,	0.0]
				]], [[
					[0.0,	2.0,	-1.0],
					[0.0,	-4.0,	2.0],
					[0.0,	2.0,	-1.0]
				]], [[
					[ 0.0,	0.0,	0.0],
					[ 2.0,	-4.0,	2.0],
					[-1.0,	2.0,	-1.0]
				]], [[
					[-1.0,	2.0,	0.0],
					[ 2.0,	-4.0,	0.0],
					[-1.0,	2.0,	0.0]
				]], [[
					[-1.0,	2.0,	-1.0],
					[ 2.0,	-4.0,	2.0],
					[-1.0,	2.0,	-1.0]
				]]]
			),
			trainable
		)
		return

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return torch.nn.functional.conv2d(x, self.weight, None, 1, 1)


class _srm_filter_5x5(torch.nn.Module):
	def __init__(self, trainable: bool = True) -> None:
		super().__init__()

		self.weight = torch.nn.Parameter(
			torch.tensor([[[
					[0.0,	0.0,	0.0,	0.0,	0.0],
					[0.0,	0.0,	0.0,	0.0,	0.0],
					[0.0,	1.0,	-3.0,	3.0,	-1.0],
					[0.0,	0.0,	0.0,	0.0,	0.0],
					[0.0,	0.0,	0.0,	0.0,	0.0]
				]], [[
					[0.0,	0.0,	0.0,	0.0,	0.0],
					[0.0,	1.0,	0.0,	0.0,	0.0],
					[0.0,	0.0,	-3.0,	0.0,	0.0],
					[0.0,	0.0,	0.0,	3.0,	0.0],
					[0.0,	0.0,	0.0,	0.0,	-1.0]
				]], [[
					[0.0,	0.0,	0.0,	0.0,	0.0],
					[0.0,	0.0,	1.0,	0.0,	0.0],
					[0.0,	0.0,	-3.0,	0.0,	0.0],
					[0.0,	0.0,	3.0,	0.0,	0.0],
					[0.0,	0.0,	-1.0,	0.0,	0.0]
				]], [[
					[ 0.0,	0.0,	0.0,	0.0,	0.0],
					[ 0.0,	0.0,	0.0,	1.0,	0.0],
					[ 0.0,	0.0,	-3.0,	0.0,	0.0],
					[ 0.0,	3.0,	0.0,	0.0,	0.0],
					[-1.0,	0.0,	0.0,	0.0,	0.0]
				]], [[
					[ 0.0,	0.0,	0.0,	0.0,	0.0],
					[ 0.0,	0.0,	0.0,	0.0,	0.0],
					[-1.0,	3.0,	-3.0,	1.0,	0.0],
					[ 0.0,	0.0,	0.0,	0.0,	0.0],
					[ 0.0,	0.0,	0.0,	0.0,	0.0]
				]], [[
					[-1.0,	0.0,	0.0,	0.0,	0.0],
					[ 0.0,	3.0,	0.0,	0.0,	0.0],
					[ 0.0,	0.0,	-3.0,	0.0,	0.0],
					[ 0.0,	0.0,	0.0,	1.0,	0.0],
					[ 0.0,	0.0,	0.0,	0.0,	0.0]
				]], [[
					[0.0,	0.0,	-1.0,	0.0,	0.0],
					[0.0,	0.0,	3.0,	0.0,	0.0],
					[0.0,	0.0,	-3.0,	0.0,	0.0],
					[0.0,	0.0,	1.0,	0.0,	0.0],
					[0.0,	0.0,	0.0,	0.0,	0.0]
				]], [[
					[0.0,	0.0,	0.0,	0.0,	-1.0],
					[0.0,	0.0,	0.0,	3.0,	0.0],
					[0.0,	0.0,	-3.0,	0.0,	0.0],
					[0.0,	1.0,	0.0,	0.0,	0.0],
					[0.0,	0.0,	0.0,	0.0,	0.0]
				]], [[
					[-1.0,	2.0,	-2.0,	2.0,	-1.0],
					[ 2.0,	-6.0,	8.0,	-6.0,	2.0],
					[-2.0,	8.0,	-12.0,	8.0,	-2.0],
					[ 0.0,	0.0,	0.0,	0.0,	0.0],
					[ 0.0,	0.0,	0.0,	0.0,	0.0]
				]], [[
					[0.0,	0.0,	-2.0,	2.0,	-1.0],
					[0.0,	0.0,	8.0,	-6.0,	2.0],
					[0.0,	0.0,	-12.0,	8.0,	-2.0],
					[0.0,	0.0,	8.0,	-6.0,	2.0],
					[0.0,	0.0,	-2.0,	2.0,	-1.0]
				]], [[
					[ 0.0,	0.0,	0.0,	0.0,	0.0],
					[ 0.0,	0.0,	0.0,	0.0,	0.0],
					[-2.0,	8.0,	-12.0,	8.0,	-2.0],
					[ 2.0,	-6.0,	8.0,	-6.0,	2.0],
					[-1.0,	2.0,	-2.0,	2.0,	-1.0]
				]], [[
					[-1.0,	2.0,	-2.0,	0.0,	0.0],
					[ 2.0,	-6.0,	8.0,	0.0,	0.0],
					[-2.0,	8.0,	-12.0,	0.0,	0.0],
					[ 2.0,	-6.0,	8.0,	0.0,	0.0],
					[-1.0,	2.0,	-2.0,	0.0,	0.0]
				]], [[
					[-1.0,	2.0,	-2.0,	2.0,	-1.0],
					[ 2.0,	-6.0,	8.0,	-6.0,	2.0],
					[-2.0,	8.0,	-12.0,	8.0,	-2.0],
					[ 2.0,	-6.0,	8.0,	-6.0,	2.0],
					[-1.0,	2.0,	-2.0,	2.0,	-1.0]
				]]]
			),
			trainable
		)
		return

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return torch.nn.functional.conv2d(x, self.weight, None, 1, 2)
