from torch.cuda import device_of
import warnings
import math
import os
import time
import torch
import pytorch_lightning as pl
import urllib.request
import numpy as np
import torch.optim as optim

from torch.utils.data import DataLoader
from tqdm import tqdm

# Torchvision
import torchvision
from torchvision.datasets import CIFAR10, CIFAR100, Places365
from torchvision import transforms

from vit_solution import VisionTransformer

# from utils.torch_utils import seed_experiment, to_device
# from utils.data_utils import save_logs
# from utils.mem_report import mem_report

"""
# Configs to run

1. python run_exp.py --model vit --layers 2 --batch_size 128 --epochs 10 --optimizer adam
2. python run_exp.py --model vit --layers 2 --batch_size 128 --epochs 10 --optimizer adamw
3. python run_exp.py --model vit --layers 2 --batch_size 128 --epochs 10 --optimizer sgd
4. python run_exp.py --model vit --layers 2 --batch_size 128 --epochs 10 --optimizer momentum

5. python run_exp.py --model vit --layers 4 --batch_size 128 --epochs 10 --optimizer adamw
6. python run_exp.py --model vit --layers 6 --batch_size 128 --epochs 10 --optimizer adamw 
7. python run_exp.py --model vit --layers 6 --batch_size 128  --epochs 10 --optimizer adamw --block postnorm

"""


# Setting the seed
pl.seed_everything(42)

# Loading the training dataset. We need to split it into a training and validation part
# We need to do a little trick because the validation set should not use the augmentation.
test_transform = transforms.Compose([transforms.Resize((160, 160)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [
                                                            0.229, 0.224, 0.225])
                                        ])
# For training, we add some augmentation. Networks are too powerful and would overfit.
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(
        (160, 160), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class ViT(pl.LightningModule):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters()
        self.train_losses, self.valid_losses = [], []
        self.train_accs, self.valid_accs = [], []
        
        # Model
        img_h, img_w = 160, 160
        num_patches = (img_h//args.patch_size) *  (img_w//args.patch_size) 
        self.model = VisionTransformer(
            embed_dim=args.hidden_dim//2,
            hidden_dim=args.hidden_dim,
            num_layers=args.layers,
            num_heads=args.num_heads,
            block=args.block,
            num_classes=10,
            patch_size=args.patch_size,
            num_patches=num_patches,
            dropout=args.dropout,
            )

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        # Optimizer
        if args.optimizer == "adamw":
            optimizer = optim.AdamW(self.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == "adam":
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
        print(
            f"Initialized {args.model} model with {sum(p.numel() for p in self.model.parameters())} "
            f"total parameters, of which {sum(p.numel() for p in self.model.parameters() if p.requires_grad)} are learnable.")

        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
        return [optimizer], [lr_scheduler]

    def _calculate_loss(self, batch, mode="train"):
        imgs, labels = batch
        logits = self.model(imgs)
        loss = self.model.loss(logits, labels)
        acc = (logits.argmax(dim=-1) == labels).float().mean()

        self.log(f'{mode}_loss', loss, on_step=False, on_epoch=True, logger=True)
        self.log(f'{mode}_acc', acc, on_step=False, on_epoch=True, logger=True)
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self._calculate_loss(batch, mode="train")
        self.train_losses.append(loss.item())
        self.train_accs.append(acc.item())
        return loss

    def training_epoch_end(self, outputs):
        train_loss = np.mean(self.train_losses)
        # self.log('train_loss', train_loss, on_step=False, on_epoch=True)
        train_acc = np.mean(self.train_accs)
        # self.log('train_acc', train_acc, on_step=False, on_epoch=True)
        if self.current_epoch % 15 == 0:
            print(f"== [TRAIN] Epoch: {self.current_epoch}, Loss: {train_loss:.3f}, Accuracy: {train_acc:.3f} ==>")        
        
        self.train_losses, self.train_accs = [], []

    def validation_step(self, batch, batch_idx):
        loss, acc = self._calculate_loss(batch, mode="val")
        self.valid_losses.append(loss.item())
        self.valid_accs.append(acc.item())
    
    def validation_epoch_end(self, outputs):
        valid_loss = np.mean(self.valid_losses)
        # self.log('valid_loss', valid_loss, on_step=False, on_epoch=True)
        valid_acc = np.mean(self.valid_accs)
        # self.log('valid_acc', valid_acc, on_step=False, on_epoch=True)        
        if self.current_epoch % 15 == 0:
            print(f"== [VAL] Epoch: {self.current_epoch}, Loss: {valid_loss:.3f}, Accuracy: {valid_acc:.3f} ==>")        
        
        self.valid_losses, self.valid_accs = [], []

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")   
    

def main(args):

    # Dataset
    if args.dataset == 'imagenette':
        num_classes = 10
        root = '/home/GRAMES.POLYMTL.CA/u114716/duke/temp/muena/nsl-project/imagenette/imagenette2-160'
    
    elif args.dataset == 'imagewoof':
        num_classes = 10
        root = '/home/GRAMES.POLYMTL.CA/u114716/duke/temp/muena/nsl-project/imagewoof/imagewoof2-160'
    else:
        raise RuntimeError("Wrong Dataset")

    full_dataset = torchvision.datasets.ImageFolder(os.path.join(root, 'train'), train_transform)

    train_size = int(0.8 * len(full_dataset))
    validation_size = len(full_dataset) - train_size
    train_set, valid_set = torch.utils.data.random_split(full_dataset, [train_size, validation_size])
    # Loading the test set
    test_set = torchvision.datasets.ImageFolder(os.path.join(root, 'val'), test_transform)

    # Dataloaders
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True, 
                                                drop_last=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=4, 
                                                        drop_last=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False,num_workers=4, 
                                                drop_last=False)


    save_path = os.path.join(os.getcwd(), "results", "imagenette")
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    wandb_logger = pl.loggers.WandbLogger(
                            # name=args.exp_id,
                            group=args.dataset, 
                            log_model=True, # save best model using checkpoint callback
                            project='nsl-project',
                            config=args)

    # to save the best model on validation
    checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=save_path,
        filename='vit_'+args.dataset, 
        monitor='val_acc', save_top_k=1, mode="max", save_last=False)
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    early_stop = pl.callbacks.EarlyStopping(monitor="val_acc", min_delta=0.00, 
                                            patience=50, verbose=False, mode="max")

    model = ViT(args)

    trainer = pl.Trainer(
        devices=args.num_gpus, accelerator="gpu", strategy="ddp",
        logger=wandb_logger, 
        callbacks=[checkpoint, lr_monitor, early_stop],
        max_epochs=args.epochs, 
        precision=32,
        enable_progress_bar=False)

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print("------- Training Done! -------")

    print("------- Loading the Best Model! ------")     # the PyTorch Lightning way
    # load the best checkpoint after training
    print(trainer.checkpoint_callback.best_model_path)

    print("------- Testing Begins! -------")
    # Test best model on validation and test set
    val_result = trainer.test(model, test_dataloaders=val_loader, verbose=False)
    test_result = trainer.test(model, test_dataloaders=test_loader, verbose=False)
    result = {"test": test_result[0]["test_acc"], "val": val_result[0]["test_acc"]}
    print(result)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run an experiment for assignment 2.")

    data = parser.add_argument_group("Data")
    
    data.add_argument("-bs", "--batch_size", type=int, default=128, help="batch size (default: %(default)s).")
    data.add_argument("--dataset", type=str, default='imagenette', help="dataset to be used (default: %(default)s).")

    model = parser.add_argument_group("Model")
    model.add_argument(
        "--model",
        type=str,
        choices=["vit"],
        default="vit",
        help="name of the model to run (default: %(default)s).",
    )
    
    model.add_argument(
        "--layers",
        type=int,
        default=2,
        help="number of layers in the model (default: %(default)s).",
    )
    model.add_argument(
        "--block",
        type=str,
        choices=["prenorm", "postnorm"],
        default='prenorm',
        help="location of LN in the encoder block (default: %(default)s).",
    )
    model.add_argument(
        "-ps", "--patch_size",
        type=int,
        default=4,
        help="the patch size to be used (default: %(default)s.",
    )
    model.add_argument(
        "-hdim", "--hidden_dim",
        type=int,
        default=384,
        help="dimension of the hidden layers (default: %(default)s.",
    )
    model.add_argument(
        "-nheads", "--num_heads",
        type=int,
        default=8,
        help="number of multihead attention (default: %(default)s.",
    )    
    model.add_argument(
        "-drp", "--dropout",
        type=float,
        default=0.0,
        help="dropout value (default: %(default)s.",
    )

    optimization = parser.add_argument_group("Optimization")
    optimization.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="number of epochs for training (default: %(default)s).",
    )
    optimization.add_argument(
        '-opt',
        "--optimizer",
        type=str,
        default="adamw",
        choices=["sgd", "momentum", "adam", "adamw"],
        help="choice of optimizer (default: %(default)s).",
    )
    optimization.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="learning rate for Adam optimizer (default: %(default)s).",
    )
    optimization.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="momentum for SGD optimizer (default: %(default)s).",
    )
    optimization.add_argument(
        "--weight_decay",
        type=float,
        default=5e-4,
        help="weight decay (default: %(default)s).",
    )

    exp = parser.add_argument_group("Experiment config")
    exp.add_argument(
        "--exp_id",
        type=str,
        default="cif100",
        help="unique experiment identifier (default: %(default)s).",
    )
    exp.add_argument(
        "--log",
        action="store_true",
        help="whether or not to log data from the experiment.",
    )
    exp.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="directory to log results to (default: %(default)s).",
    )
    exp.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed for repeatability (default: %(default)s).",
    )

    misc = parser.add_argument_group("Miscellaneous")
    misc.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="number of processes to use for data loading (default: %(default)s).",
    )
    misc.add_argument(
        "-ngpus", "--num_gpus",
        type=int,
        default=2,
        help="number of gpus (default: %(default)s).",
    )
    misc.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cuda",
        help="device to store tensors on (default: %(default)s).",
    )
    misc.add_argument(
        "--progress_bar", action="store_true", help="show tqdm progress bar."
    )
    misc.add_argument(
        "--print_every",
        type=int,
        default=10,
        help="number of minibatches after which to print loss (default: %(default)s).",
    )

    args = parser.parse_args()

    # Check for the device
    if (args.device == "cuda") and not torch.cuda.is_available():
        warnings.warn(
            "CUDA is not available, make that your environment is "
            "running on GPU (e.g. in the Notebook Settings in Google Colab). "
            'Forcing device="cpu".'
        )
        args.device = "cpu"

    if args.device == "cpu":
        warnings.warn(
            "You are about to run on CPU, and might run out of memory "
            "shortly. You can try setting batch_size=1 to reduce memory usage."
        )

    main(args)
    # # Reuse the save logs function in utils to your needs if needed.
    # if args.log is not None:
    #     save_logs(args, *logs)
