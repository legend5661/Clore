from isegm.utils.exp_imports.default import *
MODEL_NAME = 'hrnet18_selfrefiner_glas'
from isegm.data.compose import ComposeDataset,ProportionalComposeDataset
import torch.nn as nn
from isegm.data.aligned_augmentation import AlignedAugmentator
from isegm.engine.selfrefiner_trainer import ISTrainer


def main(cfg):
    model, model_cfg = init_model(cfg)
    train(model, cfg, model_cfg)


def init_model(cfg):
    model_cfg = edict()
    # model_cfg.crop_size = (256, 256)
    model_cfg.crop_size = (512, 512)
    model_cfg.num_max_points = 24
    model = HRNetModel(pipeline_version = 's2', width=18, ocr_width=48, small=False, with_aux_output=True, use_leaky_relu=True,
                       use_rgb_conv=False, use_disks=True, norm_radius=5,
                       with_prev_mask=True)

    model.to(cfg.device)
    model.apply(initializer.XavierGluon(rnd_type='gaussian', magnitude=2.0))
    # model.feature_extractor.load_pretrained_weights(cfg.IMAGENET_PRETRAINED_MODELS.HRNETV2_W18_SMALL)
    model.feature_extractor.load_pretrained_weights(cfg.IMAGENET_PRETRAINED_MODELS.HRNETV2_W18_IMAGENET) 
    #model.load_pretrained_weights('/home/admin/workspace/project/data/weights/ritm/coco_lvis_h18s_itermask.pth')
    return model, model_cfg


def train(model, cfg, model_cfg):
    cfg.batch_size = 32 if cfg.batch_size < 1 else cfg.batch_size
    cfg.val_batch_size = cfg.batch_size
    crop_size = model_cfg.crop_size

    loss_cfg = edict()
    loss_cfg.instance_loss = NormalizedFocalLossSigmoid(alpha=0.5, gamma=2)
    loss_cfg.instance_loss_weight = 1.0
    loss_cfg.instance_aux_loss = SigmoidBinaryCrossEntropyLoss()
    loss_cfg.instance_aux_loss_weight = 0.4
    loss_cfg.instance_refine_loss = NormalizedFocalLossSigmoid(alpha=0.5, gamma=2)
    loss_cfg.instance_refine_loss_weight = 1.0
    loss_cfg.instance_aux_refine_loss = SigmoidBinaryCrossEntropyLoss()
    loss_cfg.instance_aux_refine_loss_weight = 0.4
    # loss_cfg.trimap_loss = nn.BCEWithLogitsLoss() #NormalizedFocalLossSigmoid(alpha=0.5, gamma=2)
    # loss_cfg.trimap_loss_weight = 1.0

    train_augmentator = AlignedAugmentator(ratio=[0.3,1.3], target_size=crop_size,flip=True, distribution='Gaussian')
    # train_augmentator = Compose([
    #     UniformRandomResize(scale_range=(0.75, 1.40)),
    #     HorizontalFlip(),
    #     PadIfNeeded(min_height=crop_size[0], min_width=crop_size[1], border_mode=0),
    #     RandomCrop(*crop_size),
    #     RandomBrightnessContrast(brightness_limit=(-0.25, 0.25), contrast_limit=(-0.15, 0.4), p=0.75),
    #     RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.75)
    # ], p=1.0)
    
    # val_augmentator = Compose([
    #     UniformRandomResize(scale_range=(0.75, 1.25)),
    #     PadIfNeeded(min_height=crop_size[0], min_width=crop_size[1], border_mode=0),
    #     RandomCrop(*crop_size)
    # ], p=1.0)
    
    val_augmentator = Compose([
        # UniformRandomResize(scale_range=(0.75, 1.25)),
        PadIfNeeded(min_height=crop_size[0], min_width=crop_size[1], border_mode=0),
        RandomCrop(*crop_size)
    ], p=1.0)

    points_sampler = MultiPointSampler(model_cfg.num_max_points, prob_gamma=0.80,
                                       merge_objects_prob=0.15,
                                       max_num_merged_objects=2,
                                       use_hierarchy=False,
                                       first_click_center=True)

    trainset = MyGlasDataset(
        cfg.GLAS_PATH,
        csv='train.csv',
        augmentator=train_augmentator,
        min_object_area=1000,
        keep_background_prob=0.05,
        points_sampler=points_sampler,
        epoch_len=30000
    )

    valset = MyGlasDataset(
        cfg.GLAS_PATH,
        csv='val.csv',
        augmentator=val_augmentator,
        min_object_area=1000,
        points_sampler=points_sampler,
        epoch_len=2000
    )

    optimizer_params = {
        'lr': 5e-4, 'betas': (0.9, 0.999), 'eps': 1e-8
    }

    lr_scheduler = partial(torch.optim.lr_scheduler.MultiStepLR,
                           milestones=[20, 40], gamma=0.1)
    trainer = ISTrainer(model, cfg, model_cfg, loss_cfg,
                        trainset, valset,
                        optimizer='adam',
                        optimizer_params=optimizer_params,
                        lr_scheduler=lr_scheduler,
                        checkpoint_interval=[(0, 10), (50, 5)],
                        image_dump_interval=500,
                        metrics=[AdaptiveIoU()],
                        max_interactive_points=model_cfg.num_max_points,
                        max_num_next_clicks=3)
    trainer.run(num_epochs=230)
