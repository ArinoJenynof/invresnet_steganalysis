import argparse
import pathlib
lazy import lightning, lightning.pytorch.loggers, lightning.pytorch.callbacks

lazy from .models import LitInvResNet
lazy from .datasets import SteganalysisDataModuleParam, SteganalysisDataModule


__all__ = [ "main" ]


def main() -> int:
	parser = argparse.ArgumentParser(description="Training and prediction script for InvResNet steganalysis network")
	subparser = parser.add_subparsers(title="commands", help="available commands")

	training = subparser.add_parser("training", help="run steganalysis training")
	training_group = training.add_argument_group("training options")
	training_group.add_argument("dataset_root", help="folder containing steganalysis dataset")
	training_group.add_argument("trainer_root", help="folder to put training logs and model weights")
	training_group.add_argument("--epochs", help="how many epochs training will be run", default=100, type=int)

	dataset_group = training.add_argument_group("dataset options")
	dataset_group.add_argument(
		"--steganalysis-method", help="steganalysis method for training data",
		default="WOW", choices=["WOW", "S_UNIWARD"]
	)
	dataset_group.add_argument(
		"--steganalysis-payload", help="payload for training data",
		default="0.4bpp", choices=["1.0bpp", "0.8bpp", "0.6bpp", "0.4bpp", "0.2bpp"]
	)
	dataset_group.add_argument(
		"--downscaling-method", help="downscaling method for training data",
		default="Bicubic", choices=["Nearest", "Bicubic"]
	)
	dataset_group.add_argument("--pair-count", help="how many pairs in a given batch", default=1, type=int)
	dataset_group.add_argument(
		"--cacheable", help="cache loaded images in tensor format", 
		action=argparse.BooleanOptionalAction, default=True
	)
	
	training.set_defaults(func=_train_fn)

	predict = subparser.add_parser("predict", help="run prediction on an image file")
	predict.add_argument("image_file", help="image file to predict")
	predict.add_argument("model_path", help="path to model weights")
	predict.set_defaults(func=_predict_fn)

	args = parser.parse_args()
	args.func(args)
	return 0


def _train_fn(args) -> int:
	print(args)
	ds_params = SteganalysisDataModuleParam(
		args.steganalysis_method, args.steganalysis_payload.replace(".", ","),
		args.downscaling_method, args.pair_count, args.cacheable
	)
	dm = SteganalysisDataModule(args.dataset_root, ds_params)

	logger = lightning.pytorch.loggers.CSVLogger(
		args.trainer_root,
		f"{ds_params.steganalysis_method}-{ds_params.payload}-{ds_params.downscaling_method}",
	)
	loss_ckpt = lightning.pytorch.callbacks.ModelCheckpoint(
		dirpath=pathlib.Path(logger.root_dir) / "checkpoints", 
		filename="{epoch}-{val_loss:.6f}", monitor="val_loss", save_top_k=3, save_last=True
	)
	acc_ckpt = lightning.pytorch.callbacks.ModelCheckpoint(
		dirpath=pathlib.Path(logger.root_dir) / "checkpoints", 
		filename="{epoch}-{val_accuracy:.6f}", monitor="val_accuracy", save_top_k=3, mode="max"
	)
	trainer = lightning.Trainer(
		default_root_dir=args.trainer_root, logger=logger, callbacks=[loss_ckpt, acc_ckpt],
		max_epochs=args.epochs
	)

	model = LitInvResNet()
	trainer.fit(model, datamodule=dm, ckpt_path="last")
	trainer.test(model, datamodule=dm, ckpt_path="best")
	return 0


def _predict_fn(args) -> int:
	raise NotImplementedError("Not yet snake")
