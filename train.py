import argparse
from datetime import datetime, timezone, timedelta
import os, shutil, copy, random
import wandb
import random
from PIL import Image
import numpy as np

from torchvision import transforms
import torch
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from modules.losses import get_loss
from modules.optimizers import get_optimizer
from modules.parser import Parser
from modules.utils import *
from modules.dataset import CSIDataset
from modules.metrics import get_metric
from modules.earlystoppers import EarlyStopper
from modules.recorders import Recorder

from models.context_encoder import Generator, Discriminator, weights_init_normal
from models.utils import get_model

from train_cls import CLS_Train
from train_context_encoder import Inpainting_Train

import warnings
warnings.filterwarnings("ignore")

def main() :
    args = Parser().opt
        
    # Root Directory
    PROJECT_DIR = os.path.dirname(__file__)

    # Train Serial
    kst = timezone(timedelta(hours=9))
    train_serial = datetime.now(tz=kst).strftime("%Y%m%d_%H%M%S")
    
    # Recorder Directory
    if args.debug:
        RECORDER_DIR = os.path.join(PROJECT_DIR, 'results', 'train', args.output_dir, 'debug')
        if os.path.exists(RECORDER_DIR): shutil.rmtree(RECORDER_DIR)
    else:
        RECORDER_DIR = os.path.join(PROJECT_DIR, 'results', 'train',  args.output_dir, train_serial)
    
    os.makedirs(RECORDER_DIR, exist_ok=True)
    
    wandb_run_name = args.output_dir+'_'+train_serial
    
    if args.wandb:
        wandb.init(
                project=os.path.basename(os.getcwd()),
                name=wandb_run_name,
                config=args,
               )
    else:
        wandb = None
        
    # Seed
    torch_seed(args.seed)

    # CUDA setting
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    num_gpus = torch.cuda.device_count()
    visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', None)

    # Device assignments based on available GPUs and visible devices
    if cuda_available and visible_devices:
        # Parse visible devices and count them
        visible_gpus = [int(gpu) for gpu in visible_devices.split(',') if gpu.strip().isdigit()]
        num_visible_gpus = len(visible_gpus)

        if num_visible_gpus > 1:
            device_gan = torch.device(f'cuda:{visible_gpus[0]}')  # Assign first GPU for GAN
            device_cls = torch.device(f'cuda:{visible_gpus[1]}')  # Assign second GPU for ViT
        elif num_visible_gpus == 1:
            # device_gan = device_cls = torch.device(f'cuda:{visible_gpus[0]}')  # Single GPU scenario
            device_gan = device_cls = torch.device('cuda:0')  # Single GPU scenario
  # Single GPU scenario
        else:
            device_gan = device_cls = torch.device('cpu')  # Fallback to CPU if no visible GPUs
    elif cuda_available and num_gpus > 0:
        # Fallback to default CUDA device assignment if no specific visible devices set
        if num_gpus > 1:
            device_gan = torch.device('cuda:0')  # Assign first GPU for GAN
            device_cls = torch.device('cuda:1')  # Assign second GPU for ViT
        else:
            device_gan = device_cls = torch.device('cuda:0')  # Single GPU scenario
    else:
        device_gan = device_cls = torch.device('cpu')  # No GPU available, fallback to CPU
    
    '''
    Logger
    '''
    logger = get_logger(name='train', dir_=RECORDER_DIR, stream=False)
    logger.info(f"Set Logger {RECORDER_DIR}")
    
    '''
    Dataset
    '''
    transforms_ = [
        transforms.Resize((args.img_size, args.img_size), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    
    train_dataset = CSIDataset(root=PROJECT_DIR, args=args, transforms_=transforms_, mode="train")
    val_dataset = CSIDataset(root=PROJECT_DIR, args=args, transforms_=transforms_, mode="val")
    
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    
    
    '''
    Model
    '''
    if args.is_inpainting:
        generator = Generator(
                        channels=3,
                        masked_height=train_dataset.mask_h, 
                        mask_width=train_dataset.mask_w).to(device_gan)

        discriminator = Discriminator(channels=3,
            ).to(device_gan)
        
        # Initialize weights
        generator.apply(weights_init_normal)
        discriminator.apply(weights_init_normal)
        
        # Losses
        adversarial_loss = get_loss('mse').to(device_gan)
        pixelwise_loss = get_loss('l1').to(device_gan)
        
        # Optimizers
        optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.inpainting_lr, betas=(args.inpainting_b1, args.inpainting_b2))
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.inpainting_lr, betas=(args.inpainting_b1, args.inpainting_b2))

        # Recoder
        recorder_G = Recorder(record_dir=RECORDER_DIR,
                        name='Generator',
                        model=generator,
                        optimizer=optimizer_G,
                        scheduler=None,
                        logger=logger,
                        )
    
        recorder_D = Recorder(record_dir=RECORDER_DIR,
                            name='Discriminator',
                            model=discriminator,
                            optimizer=optimizer_D,
                            scheduler=None,
                            logger=logger,
                            )
        
        visualize_images_inpainting(train_dataloader, save_dir=RECORDER_DIR, wandb=wandb)
    else :
        visualize_images(train_dataloader, save_dir=RECORDER_DIR, wandb=wandb)
        
    model_args = {
        'patch_size': (args.img_size, args.vit_patch_size),
        'img_size': (args.img_size, args.img_size),
        'num_classes': len(np.unique(train_dataset.data_y)),
    }
    
    CLS = get_model(args.cls_model, model_args).to(device_cls)
    
    if args.cls_model == 'ViT':
        visualize_patches(train_dataloader, (args.img_size, args.vit_patch_size), save_dir=RECORDER_DIR, wandb=wandb)
    
    # Losses
    loss_CLS = get_loss(args.cls_loss).to(device_cls)
    
    # Optimizers    
    optimizer_CLS = get_optimizer(optimizer_name=args.cls_optimizer)
    optimizer_CLS = optimizer_CLS(params=CLS.parameters(), lr=args.cls_lr)
    scheduler_CLS = lr_scheduler.StepLR(optimizer_CLS, step_size=args.cls_step_size, gamma=args.cls_gamma)

    # Recorder
    recorder_CLS = Recorder(record_dir=RECORDER_DIR,
                            name='Classifier',
                            model=CLS,
                            optimizer=optimizer_CLS,
                            scheduler=scheduler_CLS,
                            logger=logger,
                            )
    # Metric
    metrics = {metric_name: get_metric(metric_name) for metric_name in args.metric}

    # Early stoppper
    early_stopper = EarlyStopper(patience=args.early_stopping_patience, 
                                 mode=args.early_stopping_mode,
                                 change_rate=args.early_change_rate,
                                 logger=logger)
    
    # save train args
    save_json(os.path.join(RECORDER_DIR, 'train_args.json'), vars(args))
      
    if args.wandb:
        wandb.config.update(args)
    
    if args.is_inpainting:
        trainer = Inpainting_Train([CLS, generator, discriminator],
                                   [optimizer_CLS, optimizer_G, optimizer_D],
                                   [scheduler_CLS],
                                   [loss_CLS, adversarial_loss, pixelwise_loss],
                                   [device_cls, device_gan],
                                   logger,
                                   metrics,
                                   train_dataset.patch,
                                   args.log_interval)
    else :
        trainer = CLS_Train(CLS, 
                            optimizer_CLS, 
                            scheduler_CLS,
                            loss_CLS, 
                            device_cls, 
                            logger,
                            metrics,
                            args.log_interval
                            )
    
    logger.info(f"============ TRAIN ============")
    for epoch in range(args.epochs):
        row_dict = {'epoch_index': epoch, 'train_serial': train_serial}
        
        logger.info(f"Train {epoch}/{args.epochs}")
    
        if args.is_inpainting:
            trainer.train(train_dataloader, epoch)
            row_dict['train_adv_loss'] = trainer.adv_loss_mean
            row_dict['train_pixel_loss'] = trainer.pixel_loss_mean
            
            if epoch % args.save_interval == 0:
                save_sample_images(val_dataloader, trainer, epoch, RECORDER_DIR, wandb)
        else :
            trainer.train(train_dataloader)
            
        row_dict['train_loss'] = trainer.loss_mean
            
        for metric_str, score in trainer.score_dict.items():
            row_dict[f"train_{metric_str}"] = score        
    
        if args.wandb:
            wandb.log(row_dict)
            
        trainer.clear_history()
        
        # Log results on the local
        recorder_CLS.add_row(row_dict)
        recorder_CLS.save_plot(args.plot)
        
        """
        Early stopper
        """
        if args.early_stopping : 
            early_stopping_target = args.early_stopping_target
            early_stopper.check_early_stopping(loss=row_dict[early_stopping_target])
            
            if (early_stopper.patience_counter == 0) or (epoch == args.epochs-1):
                recorder_CLS.save_weight(epoch)
                
                if args.is_inpainting:
                    recorder_G.save_weight(epoch)
                    recorder_D.save_weight(epoch)
            
            if early_stopper.stop == True:
                logger.info(f"Early Stopping at {epoch} epoch")
                break
    
    logger.info(f"============ TRAIN FINISHED ============")
    
    if args.wandb:
        wandb.finish() 
                
if __name__ == '__main__':
    main()