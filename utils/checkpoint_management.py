from pathlib import Path

import torch

from utils.common import printshare

def load_checkpoint(model, optimizer, scheduler, checkpoint_path: None | Path, hyperparams_override):

    epoch = 0
    if isinstance(checkpoint_path, Path):
        printshare("Loading pretrained model, optimizer & scheduler state dicts...")
        checkpoint = torch.load(str(checkpoint_path))

        if 'model' not in checkpoint:
            missing, unexpected = model.load_state_dict(checkpoint, strict=False)
            printshare("got no optimizer & scheduler state dicts. model state dict set up successfully.")

        else:
            missing, unexpected = model.load_state_dict(checkpoint['model'], strict=False)
            optimizer.load_state_dict(checkpoint['optimizer'])
            for g in optimizer.param_groups:
                g['weight_decay'] = hyperparams_override["optimizer_weight_decay"]

            #scheduler.load_state_dict(checkpoint["scheduler"])
            scheduler.last_epoch = checkpoint['epoch']
            epoch = checkpoint['epoch'] + 1

            printshare("all the dicts set up successfully.")


        printshare(f"[DEBUG] model missing statedict vals: {missing};")
        printshare(f"[DEBUG] model unexpected statedict vals: {unexpected}")

    return model, optimizer, scheduler, epoch


def save_checkoint_and_stats(model, optimizer, scheduler, stats,
                              checkpoints_dir: Path, stats_dir: Path):
     
    info_str = f"/ep_{stats["epoch"]+1}_p_{round(stats["val_precision"], 1)}_r_{round(stats["val_recall"], 1)}_a_{round(stats["val_accuracy"], 1)}"

    torch.save({ # model
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': stats["epoch"],

    }, checkpoints_dir / f'{info_str}_model.pth')

    torch.save(stats, #stats
        stats_dir / f'{info_str}_stats.pth')