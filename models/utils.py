import torch
import torch.nn as nn

from torchvision.models import vit_b_16, vit_b_32
from torchvision.models import swin_b
from torchvision.models import resnet50
from torchvision.models import efficientnet_b4

# 수정된 _process_input 메서드
def _process_input(self, x: torch.Tensor) -> torch.Tensor:
    n, c, h, w = x.shape  # 입력 이미지 크기
    p_h, p_w = self.patch_size  # 패치의 가로, 세로 크기

    # 비정형 이미지 크기를 지원하도록 수정
    n_h = h // p_h  # 세로 방향 패치 수
    n_w = w // p_w  # 가로 방향 패치 수

    # 패치 임베딩 적용
    x = self.conv_proj(x)

    # 패치를 펼쳐서 (n, hidden_dim, n_h * n_w) 형태로 변환
    x = x.reshape(n, self.hidden_dim, n_h * n_w)

    # Self-attention 레이어는 (N, S, E) 형식을 기대하므로 변환
    x = x.permute(0, 2, 1)

    return x

def get_model(model_name: str, model_args: dict):
    if model_name == 'ViT':
        
        patch_size = model_args['patch_size']
        img_size = model_args['img_size']
        num_classes = model_args['num_classes']
        
        vit = vit_b_16(pretrained=False)
        
        vit.patch_size = patch_size
        vit.image_size = img_size
        vit.num_classes = num_classes
        
        vit.conv_proj = nn.Conv2d(3, vit.conv_proj.out_channels, kernel_size=patch_size, stride=patch_size, padding=0)
        
        vit._process_input = _process_input.__get__(vit)
        
        # Calculate number of patches for custom image and patch sizes
        num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])  # Custom patch calculation
        num_tokens = num_patches + 1  # +1 for the class token
        
        # Resize the positional embedding to match the number of tokens (patches + class token)
        vit.encoder.pos_embedding = nn.Parameter(torch.zeros(1, num_tokens, vit.conv_proj.out_channels))
        
        # Set the number of classes for the classification task
        in_features = vit.heads[-1].in_features  # Access the last Linear layer in Sequential
        vit.heads[-1] = nn.Linear(in_features, num_classes)  # Modify the output laye      
        return vit
    
    elif model_name == 'SwinTransformer':
        # Load the Swin Transformer from torchvision
        swin_transformer = swin_b(pretrained=False)

        # Modify the classifier to match the number of classes
        in_features = swin_transformer.head.in_features  # Swin Transformer uses a single head Linear layer
        swin_transformer.head = nn.Linear(in_features, model_args['num_classes'])

        return swin_transformer
    
    elif model_name == 'ResNet50':
        resnet = resnet50(pretrained=False)
        
        # 마지막 레이어를 분류할 클래스 수에 맞게 수정
        in_features = resnet.fc.in_features
        resnet.fc = nn.Linear(in_features, model_args['num_classes'])  # 클래스 수 설정
        
        return resnet
    
    elif model_name == 'EfficientNetB4': 
        efficientnet_b4_model = efficientnet_b4(pretrained=False)
        
        # in_features = efficientnet_b4_model.classifier.in_features
        # efficientnet_b4_model.classifier = nn.Linear(in_features, model_args['num_classes'])
        
        # EfficientNet's final layer is a classifier, which we will replace
        num_ftrs = efficientnet_b4_model.classifier[1].in_features
        efficientnet_b4_model.classifier[1] = nn.Linear(num_ftrs, model_args['num_classes'])
        
        return efficientnet_b4_model
        
    else:
        raise ValueError(f'Model name {model_name} is not valid.')
