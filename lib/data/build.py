from torch.utils.data import DataLoader

from .transform import create_transform
from .datasets import BaseDataset

def make_data_loader(cfg):
    train_transform = create_transform(cfg, is_train=True)
    val_transform = create_transform(cfg, is_train=False)

    train_dataset = BaseDataset(root_path=cfg.DATASET.ROOT_DIR, list_file=cfg.DATASET.TRAIN_SPLIT, 
                    video_length=cfg.INPUT.VIDEO_LENGTH, modality=cfg.INPUT.MODALITY,
                    sample_type=cfg.INPUT.SAMPLE_TYPE,
                    image_tmpl=cfg.INPUT.IMG_TMP if cfg.INPUT.MODALITY in ["RGB", "RGBDiff"] else cfg.INPUT.FLOW_TMP,
                    transform=train_transform)

    val_dataset = BaseDataset(root_path=cfg.DATASET.ROOT_DIR, list_file=cfg.DATASET.VALIDATION_SPLIT, 
                    video_length=cfg.INPUT.VIDEO_LENGTH, modality=cfg.INPUT.MODALITY,
                    sample_type=cfg.INPUT.SAMPLE_TYPE,
                    image_tmpl=cfg.INPUT.IMG_TMP if cfg.INPUT.MODALITY in ["RGB", "RGBDiff"] else cfg.INPUT.FLOW_TMP,
                    transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=cfg.DATALOADER.BATCH_SIZE, shuffle=True,
            num_workers=cfg.DATALOADER.NUM_WORKERS, pin_memory=True)

    val_loader = DataLoader(val_dataset, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False,
            num_workers=cfg.DATALOADER.NUM_WORKERS, pin_memory=True)

    return train_loader, val_loader