import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.mobilenetv3 import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR, CosineAnnealingLR
from torch.optim import AdamW

#import architecture_generator as ag

from torchinfo import summary


from PIL import Image
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
img_size = 224
rand_state = 44
n_classes = 26
num_workers = 12


def perform_training(net: mobilenet_v3_large, training_set, validation_set, ep, lr, decay, bs,
                     pretrained: bool | str = False):
    print("training preparation...")

    train_loader = DataLoader(training_set, batch_size=bs, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(validation_set, batch_size=bs, shuffle=True, num_workers=num_workers)

    if isinstance(pretrained, str):
        print("Loading pretrained model...")
        state_dict = torch.load(pretrained)
        net.load_state_dict(state_dict, strict=True)

    epochs = ep
    criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(
        [
            {"params": net.features[-2].parameters()},  # last residual block
            {"params": net.features[-1].parameters()},  # last conv
            {"params": net.classifier.parameters()}  # classifier
        ],
        weight_decay=decay,
    )
    # scheduler = ReduceLROnPlateau(
    #    optimizer,
    #    mode='min',
    #    threshold_mode='rel',
    #    threshold=1e-2,
    #    factor=0.1,
    #    patience=2,
    #    verbose=True
    # )

    scheduler = CyclicLR(
        optimizer=optimizer,
        mode='triangular2',
        base_lr=[2.5e-4, 5e-4, 1e-3],
        max_lr=[2.5e-2, 5e-2, 1e-1],
        step_size_up=135 * 3,  # half an epoch * n = n epochs/cycle

        cycle_momentum=False
    )

    # manual testing cycle
    # while(True):
    #    image, _ = training_set[1004]
    #    plt.imshow(image)
    #    plt.title(f"Augmented sample #0")
    #    plt.axis('off')
    #    plt.show()

    print("done.")

    for epoch in range(epochs):
        print(f"epoch {epoch + 1}/{epochs} processing...")
        train_correct, train_total = perform_training_epoch(
            net=net,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler
        )
        print(f"training done. training epoch accuracy: {round(100 * train_correct / train_total, 3)}%")

        print("processing validation phase...")
        val_targets, val_predictions = perform_validation_epoch(
            net=net,
            val_loader=val_loader,
            criterion=criterion,
            scheduler=scheduler
        )
        val_precision = round(100 * precision_score(y_true=val_targets, y_pred=val_predictions, average='macro'), 3)
        val_recall = round(100 * recall_score(y_true=val_targets, y_pred=val_predictions, average='macro'), 3)
        val_accuracy = round(100 * accuracy_score(y_true=val_targets, y_pred=val_predictions), 3)

        print(f"validation done. precision: {val_precision}%; recall: {val_recall}%; accuracy: {val_accuracy}%\n")

        torch.save(net.state_dict(), f'ep_{epoch + 1}_p_{round(val_precision, 1)}_r_{round(val_recall, 1)}.pth')
    return net


def perform_training_epoch(net: mobilenet_v3_large, train_loader, criterion, optimizer, scheduler):
    correct = 0
    total = 0
    net.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.cuda(), labels.cuda()

        optimizer.zero_grad()
        outputs = net(inputs)

        pred_vals, pred_classes = torch.max(outputs.data, 1)
        correct += (pred_classes == labels).sum().item()
        total += labels.size(0)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        scheduler.step()

    return correct, total


def perform_validation_epoch(net: mobilenet_v3_large, val_loader, criterion, scheduler):
    net.eval()
    with torch.no_grad():
        targets = []
        predictions = []
        total_val_loss = 0
        for inputs, labels in val_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = net(inputs)

            total_val_loss += criterion(outputs, labels).item()

            pred_vals, pred_classes = torch.max(outputs.data, 1)

            targets = np.concatenate([targets, labels.detach().cpu().numpy()])
            predictions = np.concatenate([predictions, pred_classes.detach().cpu().numpy()])

        return targets, predictions


def perform_testing(net: mobilenet_v3_large, testing_set, weights_file: str):
    print("performing testing...")
    testing_loader = DataLoader(testing_set, batch_size=50, shuffle=True, num_workers=num_workers)

    state_dict = torch.load(weights_file)
    net.load_state_dict(state_dict, strict=True)

    net.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        targets = []
        predictions = []
        for inputs, labels in testing_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = net(inputs)

            pred_vals, pred_classes = torch.max(outputs.data, 1)
            correct += (pred_classes == labels).sum().item()
            total += labels.size(0)

            targets = np.concatenate([targets, labels.detach().cpu().numpy()])
            predictions = np.concatenate([predictions, pred_classes.detach().cpu().numpy()])

    cm = confusion_matrix(y_true=targets, y_pred=predictions, normalize="true")
    cm = np.round(cm, 3)
    class_names = ['а', 'б', 'в', 'г', 'е', 'ж', 'и', 'к', 'л', 'м', 'н', 'о', 'п', 'р', 'с', 'т', 'у', 'ф', 'х', 'ц',
                   'ч', 'ш', 'ь', 'ю', 'я', 'і']
    cmp = ConfusionMatrixDisplay(cm, display_labels=class_names)

    ax = plt.subplot()
    plt.rcParams.update({'font.size': 6})
    label_font = {'size': '13'}
    ax.set_xlabel('Predicted labels', fontdict=label_font)
    ax.set_ylabel('Observed labels', fontdict=label_font)

    title_font = {'size': '16'}
    ax.set_title('Confusion Matrix', fontdict=title_font)
    cmp.plot(ax=ax)
    plt.show()

    print(f"Finished. Accuracy: {round(100 * correct / total, 3)}%")


def custom_loader(path):
    return Image.open(path, formats=["JPEG"])


# transform wrapper for separating val&test out of train augmentation
class TransformWrapper(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        return self.transform(img), label


if __name__ == '__main__':

    net = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)

    net.cuda(0)

    summary(net, input_size=(1,3,img_size,img_size), depth=10)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            size=(img_size, img_size),
            scale=(0.6, 1.0),
            ratio=(5.0 / 6.0, 6.0 / 5.0)
        ),
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.3,
            hue=0.15
        ),
        # padding before + centercrop after rotation => corners filled.
        transforms.Pad(padding_mode="edge", padding=math.ceil(img_size * 0.2)),
        transforms.RandomRotation(
            degrees=20,
            interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.CenterCrop(img_size),

        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder('dataset', transform=None, loader=custom_loader)
    train_idx, nontrain_idx, _, nontrain_labels = train_test_split(
        np.arange(len(dataset)),
        dataset.targets,
        test_size=0.25,
        random_state=rand_state,
        shuffle=True,
        stratify=dataset.targets)
    val_idx, test_idx, _, _ = train_test_split(
        nontrain_idx,
        nontrain_labels,
        test_size=0.4,  # 0.25*0.6 = 0.15 for val, 0.25*0.4 = 0.1 for test
        random_state=rand_state,
        shuffle=True,
        stratify=nontrain_labels)

    # Subset datasets for train, val & test; labels are saved in dataset, so they`ll be auto figured out
    train_subset = TransformWrapper(Subset(dataset, train_idx), train_transform)
    validation_subset = TransformWrapper(Subset(dataset, val_idx), test_transform)
    test_subset = TransformWrapper(Subset(dataset, test_idx), test_transform)

    # this split is nonstratified, thus negatively affects model`s perfomance.
    # train_subset, validation_subset, test_subset = random_split(dataset=dataset, lengths=(0.75, 0.15, 0.1),
    #                                                           generator=torch.Generator().manual_seed(rand_state))

    # perform_training(net, train_subset, validation_subset, ep=100, lr=1e-3, decay=0.2, bs=128, pretrained=False)
    # perform_testing(net, test_subset, weights_file='ep_55_p_95.7_r_95.3.pth')

    dt = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
    ])
    test_dataset = datasets.ImageFolder('testset', transform=dt, loader=custom_loader)
    #perform_testing(net, test_dataset, weights_file='ep_55_p_95.7_r_95.3.pth')



