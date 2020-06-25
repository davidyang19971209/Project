import argparse
import json
import os

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import BrainSegmentationDataset as Dataset
from loss import DiceLoss
from unet import UNet
from utils import  dsc
from IPython import embed

def main(args):
    makedirs(args)
    snapshotargs(args)
    device = torch.device("cpu" if not torch.cuda.is_available() else args.device)

    loader_train, loader_valid = data_loaders(args)
    loaders = {"train": loader_train, "valid": loader_valid}

    unet = UNet(in_channels=Dataset.in_channels, out_channels=Dataset.out_channels)
    # unet.apply(weights_init)
    unet.to(device)

    dsc_loss = DiceLoss()
    best_validation_dsc = 0.0

    optimizer = optim.Adam(unet.parameters(), lr=args.lr, weight_decay=1e-3)
    # optimizer = optim.Adam(unet.parameters(), lr=args.lr)

    logger = Logger(args.logs)
    loss_train = []
    loss_valid = []

    log_train = []
    log_valid = []

    validation_pred = []
    validation_true = []
    step = 0

    for epoch in tqdm(range(args.epochs), total=args.epochs):
        for phase in ["train", "valid"]:
            if phase == "train":
                unet.train()
            else:
                unet.eval()


            for i, data in enumerate(loaders[phase]):
                if phase == "train":
                    step += 1
                x, y_true = data
                x, y_true = x.to(device), y_true.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    y_pred = unet(x)
                    loss = dsc_loss(y_pred, y_true)
                    print(loss)

                    # if phase == "valid":
                    if phase == "train":
                        loss_valid.append(loss.item())
                        y_pred_np = y_pred.detach().cpu().numpy()
                        validation_pred.extend(
                            [y_pred_np[s] for s in range(y_pred_np.shape[0])]
                        )
                        y_true_np = y_true.detach().cpu().numpy()
                        validation_true.extend(
                            [y_true_np[s] for s in range(y_true_np.shape[0])]
                        )

                    if phase == "train":
                        loss_train.append(loss.item())
                        loss.backward()
                        optimizer.step()


            if phase == "valid":
                dsc, label_dsc = dsc_per_volume(
                          validation_pred,
                          validation_true,
                          # loader_valid.dataset.patient_slice_index,
                          loader_train.dataset.patient_slice_index,
                          )
                mean_dsc = np.mean(dsc)
                print(mean_dsc)
                print(np.array(label_dsc).mean(axis=0))

                if mean_dsc > best_validation_dsc:
                    best_validation_dsc = mean_dsc
                    best_label_dsc = label_dsc
                    torch.save(unet.state_dict(), os.path.join(args.weights, "unet.pt"))
                    opt = epoch
                log_valid.append(np.mean(loss_valid))
                loss_valid = []
                validation_pred = []
                validation_true = []
        log_train.append(np.mean(loss_train))
        loss_train=[]

    plt.plot(log_valid)
    plt.plot(log_train)
    plt.savefig("Test")
    print("Best validation mean DSC: {:4f}".format(best_validation_dsc))
    print(opt)


def data_loaders(args):
    dataset_train, dataset_valid = datasets(args)

    def worker_init(worker_id):
        np.random.seed(42 + worker_id)

    loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        worker_init_fn=worker_init,
    )
    loader_valid = DataLoader(
        dataset_valid,
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=args.workers,
        worker_init_fn=worker_init,
    )

    return loader_train, loader_valid

def datasets(args):
    train = Dataset(
        images_dir=args.images,
        subset="train",
        image_size=args.image_size,
    )
    valid = Dataset(
        images_dir=args.images,
        subset="validation",
        image_size=args.image_size,
    )
    return train, valid


def dsc_per_volume(validation_pred, validation_true, patient_slice_index):
    dsc_list = []
    label_dsc = []
    num_slices = np.bincount([p[0] for p in patient_slice_index])
    index = 0
    for p in range(len(num_slices)):
        y_pred = np.array(validation_pred[index : index + num_slices[p]])
        y_true = np.array(validation_true[index : index + num_slices[p]])
        overall_dsc, label = dsc(y_pred,y_true) 
        dsc_list.append(overall_dsc)
        label_dsc.append(label)
        index += num_slices[p]
    return dsc_list,label_dsc


def makedirs(args):
    os.makedirs(args.weights, exist_ok=True)
    os.makedirs(args.logs, exist_ok=True)


def snapshotargs(args):
    args_file = os.path.join(args.logs, "args.json")
    with open(args_file, "w") as fp:
        json.dump(vars(args), fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training U-Net model for segmentation of brain MRI"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="input batch size for training (default: 16)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="number of epochs to train (default: 100)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="initial learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="device for training (default: cuda:0)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="number of workers for data loading (default: 4)",
    )
    parser.add_argument(
        "--vis-images",
        type=int,
        default=200,
        help="number of visualization images to save in log file (default: 200)",
    )
    parser.add_argument(
        "--vis-freq",
        type=int,
        default=10,
        help="frequency of saving images to log file (default: 10)",
    )
    parser.add_argument(
        "--weights", type=str, default="./weights", help="folder to save weights"
    )
    parser.add_argument(
        "--logs", type=str, default="./logs", help="folder to save logs"
    )
    # parser.add_argument(
    #     "--images", type=str, default="./kaggle_3m", help="root folder with images"
    # )
    parser.add_argument(
        "--images", type=str, default="C:/Users/david/Desktop/TEST_DATASET/HGG2", help="root folder with images"
        # "--images", type=str, default="D:/Project/Data/data_2019/MICCAI_BraTS_2019_Data_Training/HGG/BraTS19_TCIA01_131_1", help="root folder with images"
        
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="target input image size (default: 256)",
    )
    parser.add_argument(
        "--aug-scale",
        type=int,
        default=0.05,
        help="scale factor range for augmentation (default: 0.05)",
    )
    parser.add_argument(
        "--aug-angle",
        type=int,
        default=15,
        help="rotation angle range in degrees for augmentation (default: 15)",
    )
    args = parser.parse_args()
    main(args)
