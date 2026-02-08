import io
from pathlib import Path
import tarfile
from PIL import Image
import torch
from torchvision import tv_tensors
from torchvision.datasets import VisionDataset
from torchvision.transforms import v2


class SteganalysisDataset(VisionDataset):
	def __init__(self, covers_path: str | Path, stegos_path: str | Path, transform: v2.Transform | None = None) -> None:
		super().__init__(transform=transform)
		self.covers_p = covers_path
		self.stegos_p = stegos_path
		with tarfile.open(covers_path) as cp:
			self.covers_m = cp.getmembers()
		with tarfile.open(stegos_path) as sp:
			self.stegos_m = sp.getmembers()
		if len(self.covers_m) != len(self.stegos_m):
			raise ValueError(f"The number of covers and stegos must match (they make a pair): Have {len(self.covers_m)} covers but {len(self.stegos_m)} stegos")
		self.covers_t = torch.empty(len(self.covers_m), 1, 256, 256, dtype=torch.float32)
		self.stegos_t = torch.empty(len(self.stegos_m), 1, 256, 256, dtype=torch.float32)
		self.cached = [False for _ in range(len(self.covers_m))]
		return
	
	def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
		if not self.cached[index]:
			with tarfile.open(self.covers_p) as cp:
				imf = cp.extractfile(self.covers_m[index])
				if imf is not None:
					imb = imf.read()
					img = Image.open(io.BytesIO(imb))
					self.covers_t[index] = tv_tensors.Image(img)
				else:
					raise RuntimeError(f"Failed to extract cover member #{index}, corrupted archive?")
			with tarfile.open(self.stegos_p) as sp:
				imf = sp.extractfile(self.stegos_m[index])
				if imf is not None:
					imb = imf.read()
					img = Image.open(io.BytesIO(imb))
					self.stegos_t[index] = tv_tensors.Image(img)
				else:
					raise RuntimeError(f"Failed to extract stego member #{index}, corrupted archive?")
			self.cached[index] = True
		
		if self.transform is not None:
			return self.transform(self.covers_t[index]), torch.tensor([0.0]), self.transform(self.stegos_t[index]), torch.tensor([1.0])
		return self.covers_t[index], torch.tensor([0.0]), self.stegos_t[index], torch.tensor([1.0])
	
	def __len__(self) -> int:
		return len(self.covers_m)
