from argparse import ArgumentParser
from pathlib import Path


def main() -> int:
	parser = ArgumentParser(prog="steganalysis", description="steganalysis training script")
	parser.add_argument("dataset_root", help="Folder containing steganalysis dataset")
	parser.add_argument("trainer_root", help="Folder to put training logs and model weights")
	parser.add_argument("-sm", "--steganalysis-method", help="Steganalysis method for training data", default="WOW", choices=["WOW", "S-UNIWARD"])
	parser.add_argument("-sp", "--steganalysis-payload", help="Payload for training data", default="0.4bpp", choices=["1.0bpp", "0.8bpp", "0.6bpp", "0.4bpp", "0.2bpp"])
	parser.add_argument("-dm", "--downscaling-method", help="Downscaling method for training data", default="Bicubic", choices=["Nearest", "Bicubic"])
	parser.add_argument("-pc", "--pair-count", help="How many pairs in a given batch", default=10, type=int)
	args = parser.parse_args()

	# Delay import so printing help doesn't import everything
	from lightning import Trainer
	from lightning.pytorch.loggers import CSVLogger
	from lightning.pytorch.callbacks import ModelCheckpoint

	from steganalysis.lightning_module import DatasetParameter, LitInvResNet, SteganalysisDataModule

	ds_params = DatasetParameter(args.steganalysis_method, args.steganalysis_payload.replace(".", ","), args.downscaling_method, args.pair_count)
	dm = SteganalysisDataModule(args.dataset_root, ds_params)

	logger = CSVLogger(Path(args.trainer_root) / "logs", f"{ds_params.steganalysis_method}-{ds_params.payload}-{ds_params.downscaling_method}", 0)
	loss_ckpt = ModelCheckpoint(filename="{epoch}-{val_loss:.6f}", monitor="val_loss", save_top_k=3, save_last=True)
	acc_ckpt = ModelCheckpoint(filename="{epoch}-{val_accuracy:.6f}", monitor="val_accuracy", save_top_k=3, mode="max")
	trainer = Trainer(logger=logger, callbacks=[loss_ckpt, acc_ckpt])

	model = LitInvResNet()
	trainer.fit(model, datamodule=dm, ckpt_path="last")
	trainer.test(model, datamodule=dm, ckpt_path="best")
	return 0


if __name__ == "__main__":
	try:
		ret = main()
	except:
		raise
	else:
		raise SystemExit(ret)
