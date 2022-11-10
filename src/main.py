#!/usr/bin/env python

import argparse
import logging
import logging.config
import os
import sys
import pickle
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from plots import plot_loss_curve
from utils import train, evaluate, save_checkpoint
import random
import mymodels
import datetime
import numpy as np
from torchvision import transforms
from PIL import Image
from torch.utils.data import Subset
from torchvision.utils import save_image
from tqdm import tqdm
from data import Cifar, IMBALANCECIFAR10

logging.config.fileConfig(
    "logging.ini",
    disable_existing_loggers=False,
    defaults={
        "logfilename": datetime.datetime.now().strftime(
            "../logs/model_monitoring_%H_%M_%d_%m.log"
        )
    },
)

logger = logging.getLogger(__name__)
logger.debug("Debug Mode is enabled")

random.seed(1)
np.random.seed(1)


def predict_model(best_model, images_file, labels_file, pixel_classes):
    pass


def train_model(model_name="Vanilla_Dense"):

    logger.info("Generating DataLoader")
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    logger.info("Training Model")

    if model_name == "Vanilla_Dense":
        model = mymodels.Vanilla_Dense(3072, 256, 10)
        save_file = "Vanilla_Dense.pth"
    else:
        sys.exit("Model Not Available")

    model.to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=SGD_MOMENTUM)
    if MODEL_PATH != None:
        checkpoint = torch.load(MODEL_PATH)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        logger.info(f"Loaded Checkpoint from {MODEL_PATH}")

    logger.info(model)
    early_stopping_counter = 0

    criterion = nn.CrossEntropyLoss()
    criterion.to(DEVICE)
    train_loss_history = []
    valid_loss_history = []
    best_validation_loss = float("inf")
    for epoch in range(1, NUM_EPOCHS + 1):
        logger.info(f"Epoch {epoch}")
        train_loss = train(
            logger, model, DEVICE, train_loader, criterion, optimizer, epoch
        )
        logger.info(f"Average Loss for epoch {epoch} is {train_loss}")
        train_loss_history.append(train_loss)
        valid_loss = evaluate(logger, model, DEVICE, valid_loader, criterion, optimizer)
        valid_loss_history.append(valid_loss)
        is_best = best_validation_loss > valid_loss
        logger.info(
            f"Current Epoch loss is better : {is_best} \t Old Loss: {best_validation_loss}  vs New loss:{valid_loss}"
        )
        if epoch % EPOCH_SAVE_CHECKPOINT == 0:
            logger.info(f"Saving Checkpoint for {model_name} at epoch {epoch}")
            save_checkpoint(
                logger, model, optimizer, save_file + "_" + str(epoch) + ".tar"
            )

        if is_best:
            early_stopping_counter = 0
            logger.info(
                f"New Best Identified: \t Old Loss: {best_validation_loss}  vs New loss:\t{valid_loss} "
            )
            best_validation_loss = valid_loss
            torch.save(model, "./best_model.pth", _use_new_zipfile_serialization=False)
        else:
            logger.info("Loss didnot improve")
            early_stopping_counter += 1
        if early_stopping_counter >= PATIENCE:
            break

    # final checkpoint saved
    save_checkpoint(logger, model, optimizer, save_file + ".tar")
    # Loading Best Model
    best_model = torch.load("./best_model.pth")

    # plot loss curves
    logger.info(f"Plotting Charts")
    logger.info(f"Train Losses:{train_loss_history}")
    logger.info(f"Validation Losses:{valid_loss_history}")
    plot_loss_curve(
        logger,
        model_name,
        train_loss_history,
        valid_loss_history,
        "Loss Curve",
        f"{PLOT_OUTPUT_PATH}loss_curves.jpg",
    )
    logger.info(f"Training Finished for {model_name}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Dense Network - CIFAR10",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--gpu", action="store_true", default=True, help="Use GPU for training"
    )
    parser.add_argument(
        "--train", action="store_true", default=True, help="Train Model"
    )
    parser.add_argument(
        "--batch_size",
        nargs="?",
        type=int,
        default=32,
        help="Batch size for training the model",
    )
    parser.add_argument(
        "--num_workers", nargs="?", type=int, default=5, help="Number of Available CPUs"
    )
    parser.add_argument(
        "--num_epochs",
        nargs="?",
        type=int,
        default=10,
        help="Number of Epochs for training the model",
    )
    parser.add_argument(
        "--num_output_classes",
        nargs="?",
        type=int,
        default=10,
        help="Number of output classese for CIFAR",
    )
    parser.add_argument(
        "--learning_rate",
        nargs="?",
        type=float,
        default=0.01,
        help="Learning Rate for the optimizer",
    )
    parser.add_argument(
        "--sgd_momentum",
        nargs="?",
        type=float,
        default=0.5,
        help="Momentum for the SGD Optimizer",
    )
    parser.add_argument(
        "--plot_output_path", default="./Plots_", help="Output path for Plot"
    )
    parser.add_argument("--model_path", help="Model Path to resume training")
    parser.add_argument(
        "--epoch_save_checkpoint",
        nargs="?",
        type=int,
        default=5,
        help="Epochs after which to save model checkpoint",
    )
    parser.add_argument(
        "--patience", nargs="?", type=int, default=3, help="Early stopping epoch count"
    )
    parser.add_argument(
        "--pred_model",
        default="./checkpoint_model.pth",
        help="Model for prediction; Default is checkpoint_model.pth; \
                            change to ./best_model.pth for 1 sample best model",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logger.info(args)
    global BATCH_SIZE, USE_CUDA, NUM_EPOCHS, NUM_WORKERS, LEARNING_RATE, SGD_MOMENTUM, DEVICE, PATIENCE, PRED_MODEL
    __train__ = args.train
    BATCH_SIZE = args.batch_size
    USE_CUDA = args.gpu
    NUM_WORKERS = args.num_workers
    NUM_EPOCHS = args.num_epochs
    NUM_OUTPUT_CLASSES = args.num_output_classes
    LEARNING_RATE = args.learning_rate
    SGD_MOMENTUM = args.sgd_momentum
    PLOT_OUTPUT_PATH = args.plot_output_path
    EPOCH_SAVE_CHECKPOINT = args.epoch_save_checkpoint
    MODEL_PATH = args.model_path

    PATIENCE = args.patience
    PRED_MODEL = args.pred_model
    DEVICE = torch.device("cuda" if USE_CUDA and torch.cuda.is_available() else "cpu")

    if DEVICE.type == "cuda":
        logger.info("Settings for Cuda")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if __train__:
        logger.info("Training")
        train_model()
    else:
        best_model = torch.load(PRED_MODEL)
        logger.info(f"Using {PRED_MODEL} for prediction")
        # Predict on New Images
        # predict_model(best_model, images_file, labels_file, pixel_classes)
        logger.info("Prediction Step Complete")
