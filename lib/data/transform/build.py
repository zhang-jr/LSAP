from .video_transforms import *
import torchvision

def create_transform(cfg, is_train=True):
    normalize = GroupNormalize(mean=cfg.INPUT.MEAN, std=cfg.INPUT.STD)
    if is_train:
        augmentation = torchvision.transforms.Compose([GroupMultiScaleCrop(cfg.INPUT.BASE_SIZE, [1, .875, .75, .66]),
                                        GroupRandomHorizontalFlip(is_flow=False)])
        transform = torchvision.transforms.Compose([
                        augmentation,
                        Stack(roll=(cfg.MODEL.BACKBONE in ['BNInception', 'InceptionV3', 'resnet101'])),
                        ToTorchFormatTensor(div=(cfg.MODEL.BACKBONE not in ['BNInception', 'InceptionV3', 'resnet101'])),
                        normalize,
                    ])
    else:
        transform = torchvision.transforms.Compose([
                        GroupScale(cfg.INPUT.SCALE_SIZE),
                        GroupCenterCrop(cfg.INPUT.CROP_SIZE),
                        Stack(roll=(cfg.MODEL.BACKBONE in ['BNInception', 'InceptionV3', 'resnet101'])),
                        ToTorchFormatTensor(div=(cfg.MODEL.BACKBONE not in ['BNInception', 'InceptionV3', 'resnet101'])),
                        normalize,
                    ])
    return transform