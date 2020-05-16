from yacs.config import CfgNode as CN

_C = CN()

_C.MODEL = CN()
# Using cuda or cpu for training
_C.MODEL.DEVICE = 'cuda'
_C.MODEL.DEVICE_IDS = '0, 1, 2, 3'
_C.MODEL.SEED = 1
_C.MODEL.BACKBONE = 'resnet101'
_C.MODEL.BACKBONE_TYPE = '2D'
_C.MODEL.PRETRAINED = True
_C.MODEL.PRETRAIN_PATH = ''
_C.MODEL.PRETRAIN_CHOICE = 'imagenet'
_C.MODEL.METRIC_LOSS_TYPE = 'CrossEntropyLoss'
_C.MODEL.POOLING_TYPE = 'avg'
_C.MODEL.DROPOUT = 0.5
_C.MODEL.NO_PARTIALBN = False

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
_C.INPUT.BASE_SIZE = [224, 224]
_C.INPUT.CROP_SIZE = [224, 224]
_C.INPUT.SCALE_SIZE = [256, 256]
_C.INPUT.MEAN = [0.485, 0.456, 0.406]
_C.INPUT.STD = [0.229, 0.224, 0.225]
_C.INPUT.MODALITY = 'RGB'
_C.INPUT.SAMPLE_TYPE = 'uniform'
_C.INPUT.VIDEO_LENGTH = 16
_C.INPUT.SAMPLE_RATE = 4
_C.INPUT.IMG_TMP = 'img_{:05d}.jpg'
_C.INPUT.FLOW_TMP = 'flow_{}_{:05d}.jpg'
_C.INPUT.FLIP = True

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.NAME = 'ucf101'
_C.DATASET.NUM_CLASS = 101
_C.DATASET.ROOT_DIR = '.'
_C.DATASET.TRAIN_SPLIT = './'
_C.DATASET.VALIDATION_SPLIT = './'

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 8
_C.DATALOADER.BATCH_SIZE = 128

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.OPTIMIZER_NAME = "SGD"
_C.SOLVER.LR_SCHEDULER = 'poly'
_C.SOLVER.MAX_EPOCHS = 50
_C.SOLVER.START_EPOCH = 0
_C.SOLVER.BASE_LR = 0.001
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.WEIGHT_DECAY = 5e-4
_C.SOLVER.NESTEROV = False
_C.SOLVER.USE_TRICK = True
_C.SOLVER.LR_STEP = 20
_C.SOLVER.CLIP_GRADIENT = 'none'
_C.SOLVER.NO_PARTIALBN = True


# ---------------------------------------------------------------------------- #
# attack
# ---------------------------------------------------------------------------- #
_C.ATTACKER = CN()
_C.ATTACKER.METHOD = 'FGSM'
_C.ATTACKER.TYPE = 'linf'
_C.ATTACKER.ALPHA = 0.5
_C.ATTACKER.BETA = 1.0
_C.ATTACKER.INTER = 40

# epoch number of saving checkpoints

# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
_C.TEST = CN()
_C.TEST.BATCH_SIZE = 128
_C.TEST.WEIGHT = ""

# ---------------------------------------------------------------------------- #
# checkpoint
# ---------------------------------------------------------------------------- #
_C.CHECKPOINT = CN()
_C.CHECKPOINT.RESUME = 'none'
_C.CHECKPOINT.CHECKNAME = 'video_model'
_C.CHECKPOINT.CHECKPOINT_INTERVEL = 20
_C.CHECKPOINT.NO_VAL = False
_C.CHECKPOINT.EVAL_INTERVAL = 5
_C.CHECKPOINT.FINETUNE = False
_C.CHECKPOINT.PRINT_FREQ = 20

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Path to checkpoint and saved log of trained model
#_C.OUTPUT_DIR = ""