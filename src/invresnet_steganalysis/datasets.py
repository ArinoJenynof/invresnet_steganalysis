import dataclasses
import pathlib
import tarfile
import typing

import lightning
import PIL.Image
import sklearn.utils
import torch
import torchvision, torchvision.transforms.v2, torchvision.tv_tensors


__all__ = [ "SteganalysisDataset", "SteganalysisDataModuleParam", "SteganalysisDataModule" ]


class SteganalysisDataset(torchvision.datasets.VisionDataset):
	def __init__(
		self,
		covers_path: str | pathlib.Path, stegos_path: str | pathlib.Path,
		img_transform: torchvision.transforms.v2.Transform | None = None,
		use_cache: bool = True
	) -> None:
		super().__init__(transform=img_transform)
		self.covers_path = covers_path
		self.stegos_path = stegos_path
		self.use_cache = use_cache

		with tarfile.open(self.covers_path) as ct:
			self.covers = ct.getmembers()
		with tarfile.open(self.stegos_path) as st:
			self.stegos = st.getmembers()
		if len(self.covers) != len(self.stegos):
			raise ValueError(
				"The number of covers and stegos must match (they make a pair): "
				f"Have {len(self.covers)} covers but {len(self.stegos)} stegos"
			)

		if use_cache:
			self.covers_tensor = torch.empty(len(self.covers), 1, 256, 256, dtype=torch.float32)
			self.stegos_tensor = torch.empty(len(self.stegos), 1, 256, 256, dtype=torch.float32)
			self.cached = [False for _ in range(len(self.covers))]
		return

	def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
		if self.use_cache:
			if not self.cached[index]:
				self.covers_tensor[index] = _extract_from_tar(self.covers_path, self.covers[index])
				self.stegos_tensor[index] = _extract_from_tar(self.stegos_path, self.stegos[index])
				self.cached[index] = True

			if self.transform is not None:
				return (
					self.transform(self.covers_tensor[index]), torch.tensor([0.0]),
					self.transform(self.stegos_tensor[index]), torch.tensor([1.0])
				)
			return (
				self.covers_tensor[index], torch.tensor([0.0]),
				self.stegos_tensor[index], torch.tensor([1.0])
			)

		if self.transform is not None:
			return (
				self.transform(_extract_from_tar(self.covers_path, self.covers[index])), torch.tensor([0.0]),
				self.transform(_extract_from_tar(self.stegos_path, self.stegos[index])), torch.tensor([1.0]),
			)
		return (
			_extract_from_tar(self.covers_path, self.covers[index]), torch.tensor([0.0]),
			_extract_from_tar(self.stegos_path, self.stegos[index]), torch.tensor([1.0]),
		)

	def __len__(self) -> int:
		return len(self.covers)


@dataclasses.dataclass
class SteganalysisDataModuleParam:
	steganalysis_method: typing.Literal["WOW", "S_UNIWARD"]
	payload: typing.Literal["1,0bpp", "0,8bpp", "0,6bpp", "0,4bpp", "0,2bpp"]
	downscaling_method: typing.Literal["Nearest", "Bicubic"]
	batch_size: int
	cacheable: bool


class SteganalysisDataModule(lightning.LightningDataModule):
	def __init__(self, dataset_root: str | pathlib.Path, params: SteganalysisDataModuleParam) -> None:
		super().__init__()
		self.dataset_root = pathlib.Path(dataset_root)
		self.params = params
		self.transform = torchvision.transforms.v2.Compose([
			torchvision.transforms.v2.RandomApply([
				torchvision.transforms.v2.RandomChoice([
					torchvision.transforms.v2.RandomRotation([90, 90]),
					torchvision.transforms.v2.RandomRotation([180, 180]),
					torchvision.transforms.v2.RandomRotation([270, 270])
				])
			], 0.75),
			torchvision.transforms.v2.RandomVerticalFlip(),
			torchvision.transforms.v2.RandomHorizontalFlip()
		])
		return
	
	# TODO:
	# Make other image sources loadable, currently hardcoded to load BossBase and BOWS2. 
	# Also configurable splitting, currently 40%-10%-50% for train-valid-test
	def setup(self, stage) -> None:
		bossbase_cover = self.dataset_root / f"BossBase-cover-{self.params.downscaling_method}.tar"
		bossbase_stego = self.dataset_root / f"BossBase-{self.params.steganalysis_method}-{self.params.payload}" \
			f"-{self.params.downscaling_method}.tar"
		bows2_cover = self.dataset_root / f"BOWS2-cover-{self.params.downscaling_method}.tar"
		bows2_stego = self.dataset_root / f"BOWS2-{self.params.steganalysis_method}-{self.params.payload}" \
			f"-{self.params.downscaling_method}.tar"

		dummy_ds = SteganalysisDataset(bossbase_cover, bossbase_stego)
		ds_len = len(dummy_ds)
		train_index_end = round(ds_len * 0.4)
		valid_index_end = train_index_end + round(ds_len * 0.1)
		
		shuffled_idx = sklearn.utils.shuffle(range(ds_len), random_state=42)
		if shuffled_idx is not None:
			self.shuffled = [int(i) for i in shuffled_idx]

		bossbase_tfx = SteganalysisDataset(bossbase_cover, bossbase_stego, self.transform, self.params.cacheable)
		bows2_tfx = SteganalysisDataset(bows2_cover, bows2_stego, self.transform, self.params.cacheable)
		bossbase = SteganalysisDataset(bossbase_cover, bossbase_stego, use_cache=self.params.cacheable)

		if stage == "fit":
			if shuffled_idx is not None:
				self.ds_train = torch.utils.data.ConcatDataset([
					torch.utils.data.Subset(bossbase_tfx, self.shuffled[:train_index_end]),
					bows2_tfx
				])
				self.ds_valid = torch.utils.data.Subset(bossbase, self.shuffled[train_index_end:valid_index_end])
		
		if stage == "test":
			self.ds_test = torch.utils.data.Subset(bossbase, self.shuffled[valid_index_end:])
		
		return
	
	def train_dataloader(self) -> torch.utils.data.DataLoader:
		return torch.utils.data.DataLoader(self.ds_train, self.params.batch_size, True)
	
	def val_dataloader(self) -> torch.utils.data.DataLoader:
		return torch.utils.data.DataLoader(self.ds_valid, self.params.batch_size, False)
	
	def test_dataloader(self) -> torch.utils.data.DataLoader:
		return torch.utils.data.DataLoader(self.ds_test, self.params.batch_size, False)


def _extract_from_tar(
	tar_path: str | pathlib.Path, member: tarfile.TarInfo
) -> torchvision.tv_tensors.Image:
	with tarfile.open(tar_path) as t:
		fh = t.extractfile(member)
		if fh is not None:
			img = torchvision.tv_tensors.Image(PIL.Image.open(fh))
		else:
			raise RuntimeError(f"Failed to extract {member}, corrupted archive?")
	return img
