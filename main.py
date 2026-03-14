import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW

from sklearn.metrics import precision_score, recall_score, accuracy_score


from utils.common import curr_time, printshare

from model import create_model
from augmentations import create_transforms

from utils.data_loading import VideoFramesFolderDataset, create_dataloaders
from lr_scheduler import create_scheduler
from utils.checkpoint_management import load_checkpoint, save_checkoint_and_stats

from tester import perform_testing



IMG_SIZE = 224
RAND_STATE = 44
N_CLASSES = 33
N_WORKSERS = 4

CHECKPOINTS_DIR = Path("private/checkpoints")
STATS_DIR = CHECKPOINTS_DIR / "stats"
DATA_DIR = Path("private/data")




def perform_training(model,
                     training_set:  VideoFramesFolderDataset,
                     validation_set: VideoFramesFolderDataset,
                     epochs, w_decay, batch_size, sub_batch_size,
                     lr,
                     pretrained: bool | Path = False):

    assert batch_size % sub_batch_size == 0 #screws up gradient accumulation otherwise

    printshare("training preparation...")

    train_loader, val_loader = create_dataloaders(
        training_set=training_set,
        validation_set=validation_set,
        batch_size=sub_batch_size,
        num_workers=N_WORKSERS,
        additional_scaler_statistics_file=STATS_DIR/"ep_161_p_94.3_r_93.5_a_93.9_stats.pth"
    )

    #========= optimizer, criterion, scheduler =========

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(
        params=filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=w_decay)
    scheduler = create_scheduler(optimizer)


    #========== loading checkpoint ==========
    curr_epoch = 0

    model, optimizer, scheduler, curr_epoch = load_checkpoint(
        model=model, optimizer=optimizer, scheduler=scheduler,
        checkpoint_path=pretrained if isinstance(pretrained, Path) else None,
        hyperparams_override={"optimizer_weight_decay": w_decay}
    )

    #manual testing cycle
    #from support_scripts.dataset_visualizer import infinite_visualization
    #infinite_visualization(train_set, transform=train_transform)

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    printshare("done.")

    #========== training phase itself ==========
    while curr_epoch < epochs:
        printshare(f"[{curr_time()}] epoch {curr_epoch + 1}/{epochs} processing...")
        train_targets, train_predictions, train_loss = perform_training_epoch(
            model=model,
            full_batch_size=batch_size, sub_batch_size=sub_batch_size,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        train_precision = round(100 * precision_score(y_true=train_targets, y_pred=train_predictions, average='macro'), 3)
        train_recall = round(100 * recall_score(y_true=train_targets, y_pred=train_predictions, average='macro'), 3)
        train_accuracy = round(100 * accuracy_score(y_true=train_targets, y_pred=train_predictions), 3)

        printshare(f"training done. precision: {train_precision}%; recall: {train_recall}%; accuracy: {train_accuracy}%")


        printshare(f"[{curr_time()}] processing validation phase...")
        val_targets, val_predictions, val_loss = perform_validation_epoch(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
        )
        val_precision = round(100 * precision_score(y_true=val_targets, y_pred=val_predictions, average='macro'), 3)
        val_recall = round(100 * recall_score(y_true=val_targets, y_pred=val_predictions, average='macro'), 3)
        val_accuracy = round(100 * accuracy_score(y_true=val_targets, y_pred=val_predictions), 3)

        printshare(f"validation done. precision: {val_precision}%; recall: {val_recall}%; accuracy: {val_accuracy}%\n\n")

        stats = {
            "epoch": curr_epoch,

            "val_precision": val_precision,
            "val_recall": val_recall,
            "val_accuracy": val_accuracy,

            "val_targets": val_targets,
            "val_predictions": val_predictions,
            "val_loss": val_loss,


            "train_precision": train_precision,
            "train_recall": train_recall,
            "train_accuracy": train_accuracy,
            

            "train_targets": train_targets,
            "train_predictions": train_predictions,
            "train_loss": train_loss
            }
        
        save_checkoint_and_stats(model, optimizer, scheduler, stats,
                                 CHECKPOINTS_DIR, STATS_DIR)
        curr_epoch += 1

    printshare(f"[{curr_time()}] training successfully finished.")
    return model


def perform_training_epoch(model, full_batch_size, sub_batch_size,
                           train_loader, criterion, optimizer, scheduler):
    targets = []
    predictions = []
    batch_losses = []
    model.train()

    accum_steps = math.ceil(full_batch_size / sub_batch_size)  # number of sub-batches per "big batch"

    optimizer.zero_grad()

    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.cuda(), labels.cuda()

        outputs = model(inputs)

        pred_vals, pred_classes = torch.max(outputs.data, 1)
        targets.extend(labels.detach().cpu().numpy())
        predictions.extend(pred_classes.detach().cpu().numpy())

        loss = criterion(outputs, labels)
        loss = loss / accum_steps # smoothing the magnitude
        loss.backward()
        batch_losses.append(loss.item())

        if (i + 1) % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

    scheduler.step()
    epoch_loss = sum(batch_losses) / len(batch_losses)
    return targets, predictions, epoch_loss


def perform_validation_epoch(model, val_loader, criterion):
    model.eval()
    with torch.no_grad():
        targets = []
        predictions = []
        batch_losses = []
        for inputs, labels in val_loader:
            inputs, labels = inputs.cuda(), labels.cuda()

            outputs = model(inputs)

            batch_losses.append(criterion(outputs, labels).item())

            pred_vals, pred_classes = torch.max(outputs.data, 1)

            targets.extend(labels.detach().cpu().numpy())
            predictions.extend(pred_classes.detach().cpu().numpy())

        epoch_loss = sum(batch_losses) / len(batch_losses)
        return targets, predictions, epoch_loss



if __name__ == '__main__':


    model = create_model(N_CLASSES)

    train_transform, nontrain_transform = create_transforms(img_size=IMG_SIZE)

    train_set = VideoFramesFolderDataset(DATA_DIR / 'train', per_img_transform=train_transform, production_ready=False)
    val_set = VideoFramesFolderDataset(DATA_DIR / 'val', per_img_transform=nontrain_transform, production_ready=True)
    test_set = VideoFramesFolderDataset(DATA_DIR / 'test', per_img_transform=nontrain_transform, production_ready=True)

    perform_training(model, train_set, val_set,
                     epochs=600, w_decay=1e-4, batch_size=64, sub_batch_size=4,
                     lr=1e-3, 
                     pretrained=CHECKPOINTS_DIR / f'ep_161_p_94.3_r_93.5_a_93.9_model.pth')

    perform_testing(model, batch_size=8, testing_set=train_set,
                    weights_file=CHECKPOINTS_DIR / f'ep_161_p_94.3_r_93.5_a_93.9_model.pth')



