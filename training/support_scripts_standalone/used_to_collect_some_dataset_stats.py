import os
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


from pathlib import Path

def calculate_origin_vs_total_size_info(root_dir):
    root = Path(root_dir)
    origin_count = 0
    total_count = 0
    debug = []

    for split in root.iterdir():  # train, val, test
        if split.is_dir():
            for class_dir in split.iterdir():
                if class_dir.is_dir():
                    for vid_dir in class_dir.iterdir():
                        if vid_dir.is_dir():
                            total_count += 1
                            if "_test" not in vid_dir.name and "_mirror" not in vid_dir.name:
                                origin_count += 1
                                debug.append(str(vid_dir))

    return origin_count, total_count, debug


def count_subfolders_per_class(root_dir, train_val_test_split=False):
    root = Path(root_dir)
    class_counts = {}

    if train_val_test_split:
        for split in root.iterdir():  # train, val, test
            if split.is_dir():
                for class_dir in split.iterdir():
                    if class_dir.is_dir():
                        count = sum(1 for vid in class_dir.iterdir() if vid.is_dir())
                        if class_dir.name in class_counts:
                            class_counts[class_dir.name] += count
                        else:
                            class_counts[class_dir.name] = count

    else: #just one set
        for class_dir in root.iterdir():
            if class_dir.is_dir():
                count = sum(1 for vid in class_dir.iterdir() if vid.is_dir())
                class_counts[class_dir.name] = count

    return class_counts

def hist_classcounts(class_counts):

    sorted_classes = [
        "а", "б", "в", "г", "ґ", "д", "е", "є", "ж", "з",
        "и", "і", "ї", "й", "к", "л", "м", "н", "о", "п",
        "р", "с", "т", "у", "ф", "х", "ц", "ч", "ш", "щ",
        "ь", "ю", "я"
    ]

    # Plot histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(sorted_classes, [class_counts[cls] for cls in sorted_classes])

    # display exact counts per hist batch
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2,
            height,
            str(height),
            ha="center", va="bottom"
        )

    plt.xticks(rotation=0)
    plt.xlabel("Class")
    plt.ylabel("Num instances")
    plt.title("Distribution of video samples by class (train)")
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":

    #len_origin, len_total, _ = calculate_origin_vs_total_size_info("../data")
    #print(f"origin: {len_origin}; total: {len_total}")

    train_dir = "../private/data/train"
    dataset_dir = "../private/data"
    class_counts = count_subfolders_per_class(train_dir)
    hist_classcounts(class_counts)
