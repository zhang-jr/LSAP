from .model_wrappers import VideoModelWrapper

def create_model(cfg):
    """
    a simpler wrapper that creates the train/test models
    """
        
    model = VideoModelWrapper(cfg.DATASET.NUM_CLASS, cfg.INPUT.VIDEO_LENGTH, cfg.INPUT.MODALITY,
                backbone_name=cfg.MODEL.BACKBONE, backbone_type=cfg.MODEL.BACKBONE_TYPE, agg_fun=cfg.MODEL.POOLING_TYPE, dropout=cfg.MODEL.DROPOUT, 
                partial_bn=not cfg.SOLVER.NO_PARTIALBN, pretrained=cfg.MODEL.PRETRAINED, pretrain_path=cfg.MODEL.PRETRAIN_PATH)

    return model