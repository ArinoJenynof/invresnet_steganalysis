from collections.abc import Callable
from typing import Literal
import torch
import torchvision
from torchvision.models._utils import _make_divisible


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


class InvertedResidual(torch.nn.Module):
	def __init__(
		self,
		in_channels: int,
		expanded_channels: int,
		out_channels: int,
		kernel_size: int,
		stride: int,
		use_se: bool,
		activation_layer: Callable[..., torch.nn.Module],
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


class SpatialPyramidPool2d(torch.nn.Module):
	def __init__(
		self,
		mode: Literal["max", "avg"],
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
