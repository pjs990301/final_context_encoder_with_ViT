import pickle
import json
import yaml
import torch
import random
import logging
import os
import matplotlib.gridspec as gridspec
import torch.nn.functional as F

from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

'''
File IO
'''
def save_pickle(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_json(path, obj, sort_keys=True) -> str:
    try:
        with open(path, 'w') as f:
            json.dump(obj, f, indent=4, sort_keys=sort_keys)
        msg = f"Json saved {path}"
    except Exception as e:
        msg = f"Fail to save {e}"
    return msg


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_yaml(path, obj):
    with open(path, 'w') as f:
        yaml.dump(obj, f, sort_keys=False)


def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)    


def torch_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # np.random.seed(random_seed)
    random.seed(random_seed)
    
'''
Logger
'''
def get_logger(name: str, dir_: str, stream=False) -> logging.RootLogger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(os.path.join(dir_, f'{name}.log'))

    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    if stream:
        logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger

'''
Image inpainting
'''
def unnormalize(tensor, mean, std):
    """Unnormalize a tensor image."""
    for t, m, s in zip(tensor, mean, std):
        # t.mul_(s).add_(m)
        t = t * s + m  # Out-of-place operation
    return tensor


def convert_to_grayscale(tensor):
    # 이미지 텐서를 그레이스케일로 변환하는 변환 초기화
    grayscale_transform = transforms.Grayscale(num_output_channels=1)
    # 변환 적용
    gray_tensor = grayscale_transform(tensor)
    return gray_tensor

def visualize_patches(data_loader, patch_size, save_dir="patches", wandb=None):
    """
    데이터로더에서 이미지를 가져와 패치로 나누고 시각화하는 함수
    Args:
        data_loader: PyTorch DataLoader
        patch_size: (height, width) 형태의 패치 크기
        save_dir: 시각화된 이미지를 저장할 디렉토리
        wandb: Optional, wandb 객체
    """
    img, label = next(iter(data_loader))  # 첫 번째 배치 가져오기
    img = img[0]  # 배치에서 첫 번째 이미지 사용
    label = label[0].item()  # 레이블 가져오기
    
    vmin = img.min().item()
    vmax = img.max().item()
    
    # 패치 크기
    patch_height, patch_width = patch_size
    
    # 이미지 크기
    _, height, width = img.shape  # (C, H, W)
    
    # 시각화 준비
    fig, ax = plt.subplots(figsize=(10, 10))
    # ax.imshow(img.permute(1, 2, 0).squeeze().numpy())
    grayscale_img = convert_to_grayscale(img)
    ax.imshow(grayscale_img.squeeze().numpy(), vmin=vmin, vmax=vmax)
    ax.set_title(f"Image with Patches (Label: {label})")
    
    # # 패치 격자 그리기
    # for y in range(0, height+1, patch_height):
    #     ax.plot([0, width], [y, y], color='white', linewidth=2)
    # for x in range(0, width+1, patch_width):
    #     ax.plot([x, x], [0, height], color='white', linewidth=2)
    # 패치 격자 그리기 (경계선 보정)
    for y in range(0, height + 1, patch_height):
        ax.axhline(y - 0.5, color='red', linewidth=3)  # 가로선: 보정된 위치
    for x in range(0, width + 1, patch_width):
        ax.axvline(x - 0.5, color='red', linewidth=3)  # 세로선: 보정된 위치
    
    ax.axis('off')
    
    # 저장 및 WandB 업로드
    save_path = os.path.join(save_dir, "patches_sample.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    
    if wandb is not None and wandb.run is not None:
        wandb.log({"patch_visualization": wandb.Image(save_path)})

def visualize_images(data_loader, save_dir="images", wandb=None):
    """
    데이터로더에서 img 혹은 masked_img를 불러와서 시각화 진행
    CLS만 진행하는 CASE에서 사용
    """
    img, label = next(iter(data_loader))
    num_images = len(img)
    fig, axs = plt.subplots(1, num_images, figsize=(num_images * 3, 6))
    
    vmin = img.min().item()
    vmax = img.max().item()
    
    for idx in range(num_images):
        # 원본 이미지 시각화
        axs[idx].imshow(convert_to_grayscale(img[idx]).squeeze().numpy(), vmin=vmin, vmax=vmax)
        axs[idx].set_title('Original {label}'.format(label=label[idx]))
        axs[idx].axis('off')
        
    plt.tight_layout()
    plt.savefig(f"{save_dir}/sample.png")
    
    if wandb is not None and wandb.run is not None:
        # wandb에 이미지 로그
        wandb.log({"train_img_sample": wandb.Image(f'{save_dir}/sample.png')})
        
def visualize_images_interplation(data_loader, save_dir="images", wandb=None):
    """
    데이터로더에서 img 혹은 masked_img를 불러와서 시각화 진행
    CLS만 진행하는 CASE에서 사용
    """
    img, masked_img, _, label = next(iter(data_loader))
    num_images = len(img)
    fig, axs = plt.subplots(2, num_images, figsize=(num_images * 3, 6))
    
    vmin = img.min().item()
    vmax = img.max().item()
    
    for idx in range(num_images):
        # 원본 이미지 시각화
        axs[idx].imshow(convert_to_grayscale(img[idx]).squeeze().numpy(), vmin=vmin, vmax=vmax)
        axs[idx].set_title('Original {label}'.format(label=label[idx]))
        axs[idx].axis('off')
        
        axs[idx].imshow(convert_to_grayscale(masked_img[idx]).squeeze().numpy(), vmin=vmin, vmax=vmax)
        axs[idx].set_title('Masked {label}'.format(label=label[idx]))
        axs[idx].axis('off')
        
    plt.tight_layout()
    plt.savefig(f"{save_dir}/sample.png")
    
    if wandb is not None and wandb.run is not None:
        # wandb에 이미지 로그
        wandb.log({"train_img_sample": wandb.Image(f'{save_dir}/sample.png')})

def visualize_images_inpainting(data_loader, mode='train', save_dir="images", wandb=None):
    """데이터 로더에서 이미지를 가져와 원본과 역정규화된 이미지를 시각화"""
    if mode == 'train':
        images, masked_images, _, _, label = next(iter(data_loader))  # 첫 번째 배치 가져오기
    elif mode == 'test':
        images, masked_images, _, label = next(iter(data_loader))
        
    num_images = len(images)
    fig, axs = plt.subplots(2, num_images, figsize=(num_images * 3, 6))

    vmin = images.min().item()
    vmax = images.max().item()

    for idx in range(num_images):
        # 원본 이미지 시각화
        axs[0, idx].imshow(convert_to_grayscale(images[idx]).squeeze().numpy(), vmin=vmin, vmax=vmax)
        axs[0, idx].set_title('Original {label}'.format(label=label[idx]))
        axs[0, idx].axis('off')

        # 마스킹된 이미지 시각화
        axs[1, idx].imshow(convert_to_grayscale(masked_images[idx]).squeeze().numpy(), vmin=vmin, vmax=vmax)
        axs[1, idx].set_title('Masked {label}'.format(label=label[idx]))
        axs[1, idx].axis('off')

    plt.tight_layout()
    plt.savefig(f"{save_dir}/sample_with_masked.png")

    # wandb가 활성화되었는지 확인 후 이미지 로그
    if wandb is not None and wandb.run is not None:
        # wandb에 이미지 로그
        wandb.log({"train_img_sample": wandb.Image(f'{save_dir}/sample_with_masked.png')})

def save_sample_images(dataloader, trainer, epoch, output_dir, wandb=None) : 
    """
            sample : 원본 이미지
            masked_samples : 마스크된 이미지
            gen_mask : 생성기로부터 생성된 이미지

            original_samples : 마스크된 이미지의 원본 부분
            filled_samples : 마스크된 이미지에 생성된 마스크를 적용한 이미지
    """
    samples, masked_samples, mask_coordinates, label = next(iter(dataloader))
    
    samples = samples.to(trainer.device_G)
    masked_samples = masked_samples.to(trainer.device_G)

    # Generate inpainted image
    gen_mask = trainer.G(masked_samples).to(trainer.device_G)

    filled_samples = masked_samples.clone()
    
    for i in range(samples.size(0)):
        y1 = mask_coordinates[0][i].item()
        x1 = mask_coordinates[1][i].item()
        y2 = mask_coordinates[2][i].item()
        x2 = mask_coordinates[3][i].item()

        # i번째 생성된 마스크 선택
        single_gen_mask = gen_mask[i]  # shape: [C, H, W]

        # 슬라이싱 크기에 맞게 조정
        target_shape = (y2 - y1, x2 - x1)  # 슬라이싱된 부분의 높이와 너비
        single_gen_mask_resized = F.interpolate(
            single_gen_mask.unsqueeze(0),  # 배치 차원 추가
            size=target_shape,             # 대상 크기
            mode="bilinear", align_corners=False
        ).squeeze(0)  # 다시 배치 차원 제거

        # 슬라이싱된 부분에 생성된 마스크를 채워넣기
        filled_samples[i, :, y1:y2, x1:x2] = single_gen_mask_resized.detach()
    

    # 각 샘플별 원본 부분 추출
    original_samples = samples.clone()
    for i in range(samples.size(0)):
        y1 = mask_coordinates[0][i].item()
        x1 = mask_coordinates[1][i].item()
        y2 = mask_coordinates[2][i].item()
        x2 = mask_coordinates[3][i].item()

        original_samples = samples[:, :, y1:y2, x1:x2]

    filled_samples = unnormalize(filled_samples, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]).detach()
    masked_samples = unnormalize(masked_samples, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]).detach()
    samples = unnormalize(samples, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]).detach()
    gen_mask = unnormalize(gen_mask, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]).detach()
    original_samples = unnormalize(original_samples, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]).detach()

    # Save sample
    gray_samples = convert_to_grayscale(samples).detach().squeeze().cpu().numpy()
    gray_masked_samples = convert_to_grayscale(masked_samples).detach().squeeze().cpu().numpy()
    gray_filled_samples = convert_to_grayscale(filled_samples).detach().squeeze().cpu().numpy()
    gray_original_samples = convert_to_grayscale(original_samples).detach().squeeze().cpu().numpy()
    gray_gen_mask = convert_to_grayscale(gen_mask).detach().squeeze().cpu().numpy()

    # Plot images in a grid
    num_images = gray_samples.shape[0]

    # 전체 figure를 생성하고 gridspec을 사용하여 서브플롯을 배치
    fig = plt.figure(figsize=(num_images * 2, 10))
    outer = gridspec.GridSpec(5, num_images, wspace=0.05, hspace=0.05)  # 간격을 좁게 설정

    # 각 행에 대한 레이블을 설정합니다.
    row_labels = [
        "Original Samples",
        "Masked Samples",
        "Filled Samples",
        "Original Masks",
        "Generated Masks"
    ]

    vmin = gray_samples.min().item()
    vmax = gray_samples.max().item()

    # 각 서브플롯에 이미지를 표시합니다.
    for idx in range(num_images):
        # Display original samples
        ax = plt.Subplot(fig, outer[0, idx])
        ax.imshow(gray_samples[idx], vmin=vmin, vmax=vmax)
        ax.axis('off')
        fig.add_subplot(ax)

        # Display masked samples
        ax = plt.Subplot(fig, outer[1, idx])
        ax.imshow(gray_masked_samples[idx], vmin=vmin, vmax=vmax)
        ax.axis('off')
        fig.add_subplot(ax)

        # Display filled samples
        ax = plt.Subplot(fig, outer[2, idx])
        ax.imshow(gray_filled_samples[idx], vmin=vmin, vmax=vmax)
        ax.axis('off')
        fig.add_subplot(ax)

        # Display original masks
        ax = plt.Subplot(fig, outer[3, idx])
        ax.imshow(gray_original_samples[idx], vmin=vmin, vmax=vmax)
        ax.axis('off')
        fig.add_subplot(ax)

        # Display generated masks
        ax = plt.Subplot(fig, outer[4, idx])
        ax.imshow(gray_gen_mask[idx], vmin=vmin, vmax=vmax)
        ax.axis('off')
        fig.add_subplot(ax)

    # 각 행의 첫 번째 열에 레이블을 추가합니다.
    for i in range(len(row_labels)):
        ax = plt.Subplot(fig, outer[i, 0])
        ax.text(-0.3, 0.5, row_labels[i], va='center', ha='right', fontsize=12, transform=ax.transAxes)
        ax.axis('off')
        fig.add_subplot(ax)

    os.makedirs(f"{output_dir}/train_generated", exist_ok=True)
    plt.savefig(f"{output_dir}/train_generated/{epoch}.png", bbox_inches='tight')  # Save the figure
    plt.show()
    plt.close(fig)  # Close the figure to release memory
    
    if wandb is not None and wandb.run is not None:
        # wandb에 이미지 로그
        wandb.log({"generated_sample": wandb.Image(f'{output_dir}/generated/{epoch}.png')})
        
def test_save_sample(batches_done,img, masked_img, filled_img, mask_coordinates, save_dir):
    """
        batches_done : epoch * len(test_dataloader) + batch_index
        img : 원본 이미지
        masked_img : 마스크된 이미지
        filled_img : 생성기로부터 생성된 이미지
    """
    img = unnormalize(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]).detach()
    masked_img = unnormalize(masked_img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]).detach()
    filled_img = unnormalize(filled_img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]).detach()
    
    gray_img = convert_to_grayscale(img).detach().squeeze().cpu().numpy()
    gray_masked_img = convert_to_grayscale(masked_img).detach().squeeze().cpu().numpy()
    gray_filled_img = convert_to_grayscale(filled_img).detach().squeeze().cpu().numpy()
    
    # Plot images in a grid
    num_images = gray_img.shape[0]
    
    fig = plt.figure(figsize=(num_images * 2, 10))
    outer = gridspec.GridSpec(3, num_images, wspace=0.05, hspace=0.05)  # 간격을 좁게 설정
    
    # 각 행에 대한 레이블을 설정합니다.
    row_labels = [
        "Original Samples",
        "Masked Samples",
        "Filled Samples"
    ]
    
    vmin = gray_img.min().item()
    vmax = gray_img.max().item()
    
    for idx in range(num_images):
        # Display original samples
        ax = plt.Subplot(fig, outer[0, idx])
        ax.imshow(gray_img[idx], vmin=vmin, vmax=vmax)
        ax.axis('off')
        fig.add_subplot(ax)
        
        # Display masked samples
        ax = plt.Subplot(fig, outer[1, idx])
        ax.imshow(gray_masked_img[idx], vmin=vmin, vmax=vmax)
        
        # 마스크 좌표 가져오기
        y1 = mask_coordinates[0][idx].item()
        x1 = mask_coordinates[1][idx].item()
        y2 = mask_coordinates[2][idx].item()
        x2 = mask_coordinates[3][idx].item()

        # 회색 박스 추가 (alpha로 투명도 조절 가능)
        rect = Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2,
            edgecolor='gray',
            facecolor='gray',
            alpha=1.0  # 불투명 회색
        )
        ax.add_patch(rect)
        
        ax.axis('off')       
        fig.add_subplot(ax)
       
        # Display filled samples
        ax = plt.Subplot(fig, outer[2, idx])
        ax.imshow(gray_filled_img[idx], vmin=vmin, vmax=vmax)
        ax.axis('off')
        fig.add_subplot(ax)
    
    # 각 행의 첫 번째 열에 레이블을 추가합니다.
    for i in range(len(row_labels)):
        ax = plt.Subplot(fig, outer[i, 0])
        ax.text(-0.3, 0.5, row_labels[i], va='center', ha='right', fontsize=12, transform=ax.transAxes)
        ax.axis('off')
        fig.add_subplot(ax)
    
    os.makedirs(f"{save_dir}/test_generated", exist_ok=True)
    plt.savefig(f"{save_dir}/test_generated/{batches_done}.png", bbox_inches='tight')  # Save the figure
    plt.show()
    plt.close(fig)  # Close the figure to release memory
    
    # 따로 생성된 이미지만을 저장하기 위한 코드 작성
    # 해당 코드는 이미지의 여백 없이 논문 추가를 위해 작성된 코드임
    os.makedirs(f"{save_dir}/test_generated_img/{batches_done}", exist_ok=True)
    for idx in range(num_images) :
        plt.figure(figsize=(3,3))
        plt.imshow(gray_img[idx], vmin=vmin, vmax=vmax)
        plt.axis('off')  # Remove axis
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Completely remove margins
        plt.margins(0)  # No margins at all
        plt.gca().xaxis.set_major_locator(plt.NullLocator())  # No ticks
        plt.gca().yaxis.set_major_locator(plt.NullLocator())  # No ticks
        plt.savefig(f"{save_dir}/test_generated_img/{batches_done}/{idx}.png")  # Save the figure
        plt.show()
        plt.close()
        
        plt.figure(figsize=(3,3))
        plt.imshow(gray_masked_img[idx], vmin=vmin, vmax=vmax)
        
        # 마스크 좌표 가져오기
        y1 = mask_coordinates[0][idx].item()
        x1 = mask_coordinates[1][idx].item()
        y2 = mask_coordinates[2][idx].item()
        x2 = mask_coordinates[3][idx].item()

        # 회색 박스 추가 (alpha로 투명도 조절 가능)
        rect = Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2,
            edgecolor='gray',
            facecolor='gray',
            alpha=1.0  # 불투명 회색
        )
        plt.gca().add_patch(rect)        
        
        plt.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.margins(0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(f"{save_dir}/test_generated_img/{batches_done}/{idx}_masked.png")
        plt.show()
        plt.close()
        
        plt.figure(figsize=(3,3))
        plt.imshow(gray_filled_img[idx], vmin=vmin, vmax=vmax)
        plt.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.margins(0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(f"{save_dir}/test_generated_img/{batches_done}/{idx}_filled.png")
        plt.show()
        plt.close()