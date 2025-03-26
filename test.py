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
from modules.parser import Parser, merge_args_with_train_config
from modules.utils import *
from modules.dataset import CSIDataset
from modules.metrics import get_metric
from modules.earlystoppers import EarlyStopper
from modules.recorders import Recorder

from models.context_encoder import Generator, Discriminator, weights_init_normal
from models.utils import get_model

import sys
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

def main():
    args = Parser().opt 
    input_args = sys.argv[1:]

    PROJECT_DIR = os.path.dirname(__file__)
    train_serial = args.train_serial
    kst = timezone(timedelta(hours=9))
    predict_timestamp = datetime.now(tz=kst).strftime("%Y%m%d_%H%M%S")
    predict_serial = train_serial + '_' + predict_timestamp
    
    # Recorder Directory
    if args.debug:
        PREDICT_DIR = os.path.join(PROJECT_DIR, 'results', 'predict', args.output_dir, 'debug')
        if os.path.exists(PREDICT_DIR): shutil.rmtree(PREDICT_DIR)
    else:
        PREDICT_DIR = os.path.join(PROJECT_DIR, 'results', 'predict',  args.output_dir, predict_serial)
    
    os.makedirs(PREDICT_DIR, exist_ok=True)
    
    # Recorder Directory
    RECORDER_DIR = os.path.join(PROJECT_DIR, 'results', 'train', args.train_dir, train_serial)
    
    # Train config
    train_config = load_json(os.path.join(RECORDER_DIR, 'train_args.json'))
    
    # Merge train_config and args (command-line args take precedence)
    args = merge_args_with_train_config(train_config, args, input_args)
    
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
            device_gan = device_cls = torch.device(f'cuda:{visible_gpus[0]}')  # Single GPU scenario
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
    logger = get_logger(name='test', dir_=PREDICT_DIR, stream=False)
    logger.info(f"Set Logger {PREDICT_DIR}")
    
    '''
    Dataset
    '''
    
    transforms_ = [
        transforms.Resize((train_config['img_size'], train_config['img_size']), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    
    val_dataset = CSIDataset(root=PROJECT_DIR, args=args, transforms_=transforms_, mode="test")
    
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
                        masked_height=val_dataset.mask_h, 
                        mask_width=val_dataset.mask_w).to(device_gan)
        # Load model
        generator.load_state_dict(torch.load(os.path.join(RECORDER_DIR, 'model','Generator_model.pt'))['model'])
    else :
        visualize_images(val_dataloader, PREDICT_DIR)
        
    model_args = {
        'patch_size': (args.img_size, args.vit_patch_size),
        'img_size': (args.img_size, args.img_size),
        'num_classes': len(np.unique(val_dataset.data_y)),
    }
    
    CLS = get_model(args.cls_model, model_args).to(device_cls)
    CLS.load_state_dict(torch.load(os.path.join(RECORDER_DIR, 'model','Classifier_model.pt'))['model'])
    
    # Losses
    loss_CLS = get_loss(args.cls_loss).to(device_cls)
    
    # Recorder
    recorder_CLS = Recorder(record_dir=PREDICT_DIR,
                            name='Classifier',
                            model=CLS,
                            optimizer=None,
                            scheduler=None,
                            logger=logger,
                            )
    
    # save train args
    save_json(os.path.join(PREDICT_DIR, 'test_args.json'), vars(args))
    save_json(os.path.join(PREDICT_DIR, 'train_args.json'), train_config)
    
    
    """
    val
    """
    CLS.eval()
    if args.is_inpainting:
        generator.eval()
    
    CLS_batch_loss_sum, CLS_batch_correct, CLS_batch_total = 0.0, 0, 0
    row_dict = {'epoch_index': 0, 'val_accuracy': 0, 'val_loss': 0, 
                'train_serial': train_serial, 'predict_serial': predict_serial}
    
    with torch.no_grad():
        if args.is_inpainting:
            for batch_index, (img, masked_img, mask_coordinates, label) in enumerate(tqdm(val_dataloader)):
                masked_img = masked_img.to(device_gan)
                filled_img = masked_img.clone()
                
                for i in range(masked_img.size(0)):
                    y1,x1,y2,x2 = mask_coordinates[0][i].item(), mask_coordinates[1][i].item(), mask_coordinates[2][i].item(), mask_coordinates[3][i].item()
                    gen_mask = generator(masked_img[i].unsqueeze(0)).to(device_gan)
                    
                    # 마스크 부분을 생성된 마스크로 채움
                    filled_img[i, :, y1:y2, x1:x2] = gen_mask.detach()
                
                test_save_sample(batches_done=batch_index, save_dir = PREDICT_DIR, img=img, masked_img=masked_img, filled_img=filled_img, mask_coordinates=mask_coordinates)
                                
                x = filled_img.to(device_cls, dtype=torch.float)
                y = label.to(device_cls, dtype=torch.long)
                    
                y_pred = CLS(x)
                loss = loss_CLS(y_pred, y)
                _, predicted = torch.max(y_pred.data, 1)
                
                CLS_batch_loss_sum += loss.item() * y.size(0)
                CLS_batch_total += y.size(0)
                CLS_batch_correct += (predicted == y).sum().item()
        else :
            for batch_index, (img, label) in enumerate(tqdm(val_dataloader)):

                x = img.to(device_cls, dtype=torch.float)                    
                y = label.to(device_cls, dtype=torch.long)
                
                y_pred = CLS(x)
                loss = loss_CLS(y_pred, y)
                _, predicted = torch.max(y_pred.data, 1)
                
                CLS_batch_loss_sum += loss.item() * y.size(0)
                CLS_batch_total += y.size(0)
                CLS_batch_correct += (predicted == y).sum().item()
    
    # Epoch history
    val_loss = CLS_batch_loss_sum / len(val_dataloader.dataset)
    val_accuracy = CLS_batch_correct / CLS_batch_total

    row_dict['val_loss'] = val_loss
    row_dict['val_accuracy'] = val_accuracy

    """
    Record
    """
    recorder_CLS.add_row(row_dict)
    # recorder_CLS.save_plot(args.plot)
    
    logger.info(f"Val Loss: {row_dict['val_loss']}, Accuracy: {row_dict['val_accuracy']}")
    print('Validation Loss:', val_loss, 'Validation Accuracy:', val_accuracy)
    logger.info(f"============ TEST FINISHED ============")

if __name__ == '__main__':
    main()