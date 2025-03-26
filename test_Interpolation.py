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
import torch.nn.functional as F

import sys
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

def fill_masked_image(masked_img, mode='bilinear'):
    """
    마스킹된 부분을 전체 데이터를 기준으로 보간하여 복원합니다.
    :param masked_img: 마스킹된 이미지 (Tensor, Shape: [C, H, W])
    :param mode: 보간 방식 ('nearest', 'linear', 'bilinear', 'bicubic')
    :return: 보간된 이미지 (Tensor, Shape: [C, H, W])
    """
    c, h, w = masked_img.shape

    # 마스크 정의: 값이 1인 부분을 마스킹으로 처리
    mask = (masked_img == 1).float()  # 마스킹된 영역
    non_mask = (masked_img != 1).float()  # 마스킹되지 않은 영역

    # 보간을 위해 데이터 타입을 float으로 변환
    filled_img = masked_img.clone().float()

    for channel in range(c):  # 각 채널별로 보간 수행
        img_channel = filled_img[channel]  # Shape: [H, W]
        mask_channel = mask[channel]
        non_mask_channel = non_mask[channel]

        if mode == 'linear':
            # 이미지를 (H, W)에서 (H*W,) 형태의 1차원 텐서로 변환
            img_flat = img_channel.view(1, 1, -1)  # Shape: [1, 1, H*W]
            mask_flat = mask_channel.view(1, 1, -1)
            non_mask_flat = non_mask_channel.view(1, 1, -1)

            # 마스킹되지 않은 영역은 원본 값 유지, 마스킹된 영역은 0으로 설정
            img_flat_masked = img_flat * non_mask_flat

            # 1차원 보간 수행
            interpolated_flat = F.interpolate(
                img_flat_masked,
                size=img_flat_masked.shape[-1],
                mode='linear',
                align_corners=True
            )

            # 보간된 결과를 원래 이미지 형태로 복원
            interpolated = interpolated_flat.view(h, w)  # Shape: [H, W]

        else:
            # 2D 보간 수행 (bilinear 또는 bicubic)
            img_channel = img_channel.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, H, W]
            non_mask_channel = non_mask_channel.unsqueeze(0).unsqueeze(0)

            # 마스킹되지 않은 영역은 원본 값 유지, 마스킹된 영역은 0으로 설정
            img_channel_masked = img_channel * non_mask_channel

            # 보간 수행
            interpolated = F.interpolate(
                img_channel_masked,
                size=(h, w),
                mode=mode,
                align_corners=(True if mode in ['linear', 'bilinear', 'bicubic'] else None)
            ).squeeze(0).squeeze(0)  # Shape: [H, W]

        # 마스킹된 영역에 보간된 값 채움
        filled_img[channel] = non_mask_channel * img_channel + mask_channel * interpolated

    # 데이터 타입 복원 (필요한 경우)
    filled_img = filled_img.type(masked_img.dtype)

    return filled_img


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
    
    # Device assignments based on available GPUs
    if cuda_available and num_gpus > 1:
        device_gan = torch.device('cuda:0')  # Assign first GPU for GAN
        device_cls = torch.device('cuda:1')  # Assign second GPU for ViT
    elif cuda_available and num_gpus == 1:
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
    visualize_images_inpainting(val_dataloader, save_dir=PREDICT_DIR, mode='test')    
        
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
    CLS_batch_loss_sum, CLS_batch_correct, CLS_batch_total = 0.0, 0, 0
    row_dict = {'epoch_index': 0, 'val_accuracy': 0, 'val_loss': 0, 
                'train_serial': train_serial, 'predict_serial': predict_serial}
    
    with torch.no_grad():
        if args.is_inpainting:
            for batch_index, (img, masked_img, mask_coordinates, label) in enumerate(tqdm(val_dataloader)):
                masked_img = masked_img.to(device_gan)
                filled_img = masked_img.clone()
                
                # 배치 단위 처리
                for i in range(masked_img.size(0)):
                    # 손실된 영역만 보간 수행
                    filled_img[i] = fill_masked_image(masked_img[i], mode=args.test_interpolation)  # mode 선택 가능
                
                test_save_sample(batches_done=batch_index, save_dir = PREDICT_DIR, img=img, masked_img=masked_img, filled_img=filled_img)
                                
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