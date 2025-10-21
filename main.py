import math
import os
import torch
import torch.nn as nn

from torchvision.transforms import v2
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from torch.optim import AdamW
from pytorchvideo.models.hub import x3d_s, x3d_m

from own_architecture_attempt import MobileNet3D
from video_loading_utils import VideoFramesFolderDataset, TransformSubset
from tester import perform_testing
from support_scripts.weighted_sampling_distributor import analyze_weaknesses_produce_weights

from PIL import Image
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, accuracy_score

import matplotlib.pyplot as plt
from datetime import datetime
from zoneinfo import ZoneInfo


img_size = 224
rand_state = 44
n_classes = 33
num_workers = 4


def curr_time():
    return datetime.now(ZoneInfo('Europe/Kiev'))


def printshare(msg, logfile="training_log.txt"):
    print(msg)

    with open(logfile, "a") as f:
        print(msg, file=f)


def cosannealing_decay_warmup(warmup_steps, T_0, T_mult, decay_factor, base_lr, eta_min):
    # returns the func that performs all the calculations.
    # useful for keeping all the params in one place = scheduler def.
    def lr_lambda(epoch): #0-based epoch
        if epoch < warmup_steps:
            return base_lr * ((epoch + 1) / warmup_steps)

        annealing_step = epoch - warmup_steps

        # calculating which cycle (zero-based) are we in,
        # current cycle length (T_current) and position inside the cycle (t)
        if T_mult == 1:
            cycle = annealing_step // T_0
            t = annealing_step % T_0
            T_current = T_0

        else:
            # fast log-based computation
            cycle = int(math.log((annealing_step * (T_mult - 1)) / T_0 + 1, T_mult))
            sum_steps_of_previous_cycles = T_0 * (T_mult ** cycle - 1) // (T_mult - 1)
            t = annealing_step - sum_steps_of_previous_cycles
            T_current = T_0 * (T_mult ** cycle)


        # enable decay
        eta_max = base_lr * (decay_factor ** cycle)

        # cosine schedule between (eta_min, max_lr]
        lr = eta_min + 0.5 * (eta_max-eta_min) * (1 + math.cos(math.pi * t / T_current))
        return lr/base_lr

    return lr_lambda






def perform_training(net,
                     training_set: TransformSubset | VideoFramesFolderDataset,
                     validation_set: TransformSubset | VideoFramesFolderDataset,
                     epochs, w_decay, batch_size, sub_batch_size,
                     lr, lr_lambda: cosannealing_decay_warmup,
                     pretrained: bool | str = False):

    assert batch_size % sub_batch_size == 0 #screws up gradient accumulation otherwise

    printshare("training preparation...")

    #======== creating balanced-batch dataloaders ========
    t_class_counts = np.bincount(training_set.targets)
    t_class_weights = 1.0 / t_class_counts #weights based on dataset imbalance
    t_class_modifiers = analyze_weaknesses_produce_weights() #extra modifiers based on current situation and model`s weaknesses
    t_class_weights *= t_class_modifiers

    t_sample_weights = [t_class_weights[t] for t in training_set.targets]
    train_sampler = WeightedRandomSampler(t_sample_weights, num_samples=len(t_sample_weights), replacement=True)
    train_loader = DataLoader(training_set, batch_size=sub_batch_size, sampler=train_sampler, num_workers=num_workers)

    v_class_counts = np.bincount(validation_set.targets)
    v_class_weights = 1.0 / v_class_counts
    v_sample_weights = [v_class_weights[t] for t in validation_set.targets]
    val_sampler = WeightedRandomSampler(v_sample_weights, num_samples=len(v_sample_weights), replacement=True)
    val_loader = DataLoader(validation_set, batch_size=sub_batch_size, sampler=val_sampler, num_workers=num_workers)

    #========= loading the checkpoint and preparing optimizers =========

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(
        params=filter(lambda p: p.requires_grad, net.parameters()),
        lr=lr, weight_decay=w_decay)
        #[
        #    {"params": net.features[-2].parameters()},  # last residual block
        #    {"params": net.features[-1].parameters()},  # last conv
        #    {"params": net.classifier.parameters()}  # classifier
        #],

    #used LambdaLR to implement CosineAnnealing with warm restarts and decay.
    #yup, we need the base_lr to be passed in, cause it looks like this is the safest way.
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lr_lambda
    )

    #scheduler = CosineAnnealingLR(
    #    optimizer=optimizer,
    #    T_max=50,
    #    eta_min=1e-8,
    #)

    curr_epoch = 0
    if isinstance(pretrained, str):
        printshare("Loading pretrained model, optimizer & scheduler state dicts...")
        checkpoint = torch.load(pretrained)
        mid_se_keys = ["mid_se.fc.0.weight", "mid_se.fc.0.bias", "mid_se.fc.2.weight", "mid_se.fc.2.bias"]

        if 'model' not in checkpoint:
            missing, unexpected = net.load_state_dict(checkpoint, strict=False)
            printshare("got no optimizer & scheduler state dicts. model state dict set up successfully.")

        else:
            missing, unexpected = net.load_state_dict(checkpoint['model'], strict=False)
            optimizer.load_state_dict(checkpoint['optimizer'])
            for g in optimizer.param_groups:
                g['weight_decay'] = w_decay

            #scheduler.load_state_dict(checkpoint["scheduler"])
            scheduler.last_epoch = checkpoint['epoch']
            curr_epoch = checkpoint['epoch'] + 1

            printshare("all the dicts set up successfully.")


        printshare(f"[DEBUG] model missing statedict vals: {missing};")
        printshare(f"[DEBUG] model unexpected statedict vals: {unexpected}")

    #manual testing cycle
    #while(True):

    #    image, _ = training_set[225]
    #    transform = v2.ToPILImage()
    #    for i in range(16):
    #        img = transform(image[i])
    #        plt.imshow(img)
    #        plt.title(f"Augmented sample #0")
    #        plt.axis('off')
    #        plt.show()

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("checkpoints/stats", exist_ok=True)
    printshare("done.")

    #========== training itself ==========
    while curr_epoch < epochs:
        printshare(f"[{curr_time().strftime('%Y-%m-%d %H:%M:%S')}] epoch {curr_epoch + 1}/{epochs} processing...")
        train_targets, train_predictions, train_loss = perform_training_epoch(
            net=net,
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


        printshare(f"[{curr_time().strftime('%Y-%m-%d %H:%M:%S')}] processing validation phase...")
        val_targets, val_predictions, val_loss = perform_validation_epoch(
            net=net,
            val_loader=val_loader,
            criterion=criterion,
        )
        val_precision = round(100 * precision_score(y_true=val_targets, y_pred=val_predictions, average='macro'), 3)
        val_recall = round(100 * recall_score(y_true=val_targets, y_pred=val_predictions, average='macro'), 3)
        val_accuracy = round(100 * accuracy_score(y_true=val_targets, y_pred=val_predictions), 3)

        printshare(f"validation done. precision: {val_precision}%; recall: {val_recall}%; accuracy: {val_accuracy}%\n\n")

        torch.save({ # model
            'model': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': curr_epoch,

        }, f'checkpoints/ep_{curr_epoch+1}_p_{round(val_precision, 1)}_r_{round(val_recall, 1)}_a_{round(val_accuracy, 1)}_model.pth')

        torch.save({ # stats
            'epoch': curr_epoch,
            'train_targets': train_targets,
            'train_predictions': train_predictions,
            'train_loss': train_loss,
            'val_loss': val_loss,
        },
            f'checkpoints/stats/ep_{curr_epoch + 1}_p_{round(val_precision, 1)}_r_{round(val_recall, 1)}_a_{round(val_accuracy, 1)}_stats.pth')

        curr_epoch += 1

    printshare(f"[{curr_time().strftime('%Y-%m-%d %H:%M:%S')}] training successfully finished.")
    return net


def perform_training_epoch(net: MobileNet3D, full_batch_size, sub_batch_size,
                           train_loader, criterion, optimizer, scheduler):
    targets = []
    predictions = []
    batch_losses = []
    net.train()

    accum_steps = math.ceil(full_batch_size / sub_batch_size)  # number of sub-batches per "big batch"

    optimizer.zero_grad()

    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.cuda(), labels.cuda()

        outputs = net(inputs)

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


def perform_validation_epoch(net: MobileNet3D, val_loader, criterion):
    net.eval()
    with torch.no_grad():
        targets = []
        predictions = []
        batch_losses = []
        for inputs, labels in val_loader:
            inputs, labels = inputs.cuda(), labels.cuda()

            outputs = net(inputs)

            batch_losses.append(criterion(outputs, labels).item())

            pred_vals, pred_classes = torch.max(outputs.data, 1)

            targets.extend(labels.detach().cpu().numpy())
            predictions.extend(pred_classes.detach().cpu().numpy())

        epoch_loss = sum(batch_losses) / len(batch_losses)
        return targets, predictions, epoch_loss





def custom_loader(path):
    return Image.open(path, formats=["JPEG"])




if __name__ == '__main__':

    #net = MobileNet3D(n_classes=33, width_mult=0.8)

    net = x3d_m(pretrained=True)

    #replacing the classifier head
    in_features = net.blocks[-1].proj.in_features
    net.blocks[-1].proj = nn.Linear(in_features, n_classes)

    #correcting running_stats for tiny batches
    for m in net.modules():
        if isinstance(m, nn.BatchNorm3d):
            m.momentum = 5e-4


    net.cuda(0)

    train_transform = v2.Compose([
        v2.RandomResizedCrop(
            size=(img_size, img_size),
            scale=(0.75, 1.0),
            ratio=(7.0 / 8.0, 8.0 / 7.0)
        ),
        v2.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05
        ),

        # edge padding before + centercrop after rotation => corners filled.
        v2.Pad(padding_mode="edge", padding=math.ceil(img_size * 0.2)),
        v2.RandomRotation(
            degrees=15,
            interpolation=v2.InterpolationMode.BILINEAR
        ),
        v2.CenterCrop(img_size),

        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=True),
        v2.GaussianNoise(mean=0, sigma=0.08),
        v2.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
    ])

    nontrain_transform = v2.Compose([
        v2.Resize(size=(img_size, img_size)),
        v2.ToDtype(torch.float32, scale=True),
    ])

    train_set = VideoFramesFolderDataset('data/train', per_img_transform=train_transform, production_ready=True)
    val_set = VideoFramesFolderDataset('data/val', per_img_transform=nontrain_transform, production_ready=True)
    test_set = VideoFramesFolderDataset('data/test', per_img_transform=nontrain_transform, production_ready=True)

    #perform_training(net, train_set, val_set,
    #                 epochs=600, w_decay=1e-4, batch_size=64, sub_batch_size=4,
    #                 lr=1e-3, lr_lambda=cosannealing_decay_warmup(
    #                   warmup_steps=0, T_0=10, T_mult=1.1, decay_factor=0.9, base_lr=1e-3, eta_min=1e-8),
    #                 pretrained='checkpoints/ep_172_p_90.0_r_89.1_a_88.8_model.pth')

    perform_testing(net=net, batch_size=8, testing_set=train_set,
                    weights_file='checkpoints/ep_161_p_94.3_r_93.5_a_93.9_model.pth')



