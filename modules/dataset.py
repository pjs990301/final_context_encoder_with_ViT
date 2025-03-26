import os
import glob

import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image

import pandas as pd
import numpy as np
from sklearn.utils import shuffle

from modules.utils import *
from modules.preprocessing import *

null_subcarrier = [
    '_' + str(x + 32) for x in [-32, -31, -30, -29, -28, -27, 0, 27, 28, 29, 30, 31]]
pilot_subcarrier = ['_' + str(x + 32) for x in [-21, -7, 7, 21]]
additional_subcarrier = ['_' + str(x + 32) for x in [-1, 1, -26, 26]]
unnecessary_columns = ['mac', 'time']


class CSIDataset(Dataset) :
    def __init__(self, root, args, transforms_, mode="train") :
        '''
            root : str : 데이터셋이 저장된 디렉토리
            args : argparse : 사용자가 입력한 인자
            transforms : torchvision.transforms : 데이터셋에 적용할 전처리
            mode : str : train 또는 test           
        '''
        self.root = root
        self.args = args
        self.mode = mode
        self.radio = 0.8
        self.transform = transforms.Compose(transforms_)
        self.mapping = load_yaml('config/mapping.yaml')
        self.data, self.subcarrier_list, self.min_data_len, self.number_of_RPI = self.load_csi_from_csv()
        self.data = self.slice_train_test()
        
        if self.args.is_preprocess :
            self.data = preprocess(self.data, self.args)
        
        if self.args.is_split:
            self.data = time_mod(self.data, self.args, self.mapping)
                
        if self.args.is_PCA :
            self.data = pca(self.data, self.args)
        
        self.data_x, self.data_y = self.generate_windows()

        # # data_x와 data_y를 랜덤하게 섞음
        self.data_x, self.data_y = shuffle(self.data_x, self.data_y, random_state=self.args.seed)
        
        # # train : validation = 8 : 2
        # if self.mode == "train":
        #     self.data_x, self.data_y = self.data_x[:int(len(self.data_x) * 0.8)], self.data_y[:int(len(self.data_y) * 0.8)]
        # else:
        #     self.data_x, self.data_y = self.data_x[int(len(self.data_x) * 0.8):], self.data_y[int(len(self.data_y) * 0.8):]
        
        if self.args.is_inpainting or self.args.test_mask:
            self.img_h, self.img_w = self.args.img_size, self.args.img_size
            if self.args.use_mask :
                self.mask_h = self.args.mask_h
                self.mask_w = self.args.mask_w
            else:
                self.mask_h = int(self.data_x.shape[1] * (self.args.img_size / self.data_x.shape[1]))
                self.mask_w = int(self.args.round_robin_size* (self.args.img_size / self.args.window_size))
        
            # Patch 크기 조정 (마스크 크기에 맞춰)
            patch_h, patch_w = int(self.mask_h / 2 ** 3), int(self.mask_w / 2 ** 3)
            self.patch = (1, patch_h, patch_w)
            
            
    def load_csi_from_csv(self) :
        min_data_len = int(1e9)
        data = {}
        subcarrier_list = []
        dir_path = os.path.join(self.root, 'data', self.args.dataset, 
        # dir_path = os.path.join('data', self.root, self.args.dataset, 
                            self.args.csi_type, 
                            str(self.args.enviroment)+"_"+
                            self.mapping[self.args.dataset]['path'].get(str(self.args.RPI))
                            )
        for dir in os.listdir(dir_path):
            files = os.listdir(os.path.join(dir_path, dir))
            for file in files:
                
                parts = file.split('_')
                location, label, RPI = parts[0], parts[1], parts[2]
                data.setdefault(location, {}).setdefault(label, {}).setdefault(RPI, {})
                csv_data = pd.read_csv(os.path.join(dir_path, dir, file), engine='pyarrow')
                
                if self.args.is_sampling:
                    csv_data = csv_data.iloc[::self.args.sampling_rate]

                if self.args.is_exclude:
                    excluded_columns = additional_subcarrier + null_subcarrier + pilot_subcarrier
                    csv_data = csv_data.drop(excluded_columns, axis=1, errors='ignore').reset_index(drop=True)
                min_data_len = min(min_data_len, len(csv_data))
                data[location][label][RPI] = csv_data

                # 서브캐리어 목록만을 추출
                subcarrier_list = [column for column in csv_data.columns.tolist() if column not in unnecessary_columns]
                
        #  모든 RPI 데이터에 대해서 길이가 동일할 수 있도록 보장
        for location in data:
            for label in data[location]:
                for RPI in data[location][label]:
                    slice_data = data[location][label][RPI]
                    data[location][label][RPI] = slice_data.iloc[:min_data_len]
        number_of_RPI = len(self.args.RPI)
        
        return data, subcarrier_list, min_data_len, number_of_RPI     
    
    def slice_train_test(self) :
        data = self.data
        mode = self.mode
        radio = self.radio
       
        for location in data:
            for label in data[location]:
                for RPI in data[location][label]:
                    # 현재 RPI 데이터
                    df = data[location][label][RPI]

                    # 슬라이싱
                    if mode == "train":
                        data[location][label][RPI] = df.iloc[:int(len(df) * radio)].reset_index(drop=True)
                    else:
                        data[location][label][RPI] = df.iloc[int(len(df) * radio):].reset_index(drop=True)
                        
        for location in data:
            for label in data[location]:
                for RPI in data[location][label]:
                    df_len = len(data[location][label][RPI])
                    if df_len > 0:  # 빈 데이터는 무시
                        self.min_data_len = min(self.min_data_len, df_len)
                    
        return data
    
    def generate_windows(self):
        data_x, data_y = [], []

        min_num_wd = self.min_data_len // (self.args.window_size)

        for location in self.data:
            for label in self.data[location]:
                if self.args.is_split:
                    for i in range(min_num_wd):
                        window = self.data[location][label].iloc[i * self.args.window_size:i * self.args.window_size + self.args.window_size]
                        data_x.append(window)
                        label_vector = self.mapping[self.args.dataset]["Location_Label"][f"{location}_{label}"]
                        data_y.append(label_vector)
                else:
                    for RPI in self.data[location][label]:
                        selected_data = self.data[location][label][RPI]
                        selected_data = selected_data.drop(unnecessary_columns, axis=1).reset_index(drop=True)

                        for i in range(min_num_wd):
                            window = selected_data.iloc[i * self.args.window_size:i * self.args.window_size + self.args.window_size].values
                            data_x.append(window)
                            label_vector = self.mapping[self.args.dataset]["Location_Label"][f"{location}_{label}"]
                            data_y.append(label_vector)
                            
        data_x = np.array(data_x)
        data_x = np.transpose(data_x, (0, 2, 1))       
       
        return data_x, np.array(data_y)

    def __getitem__(self, idx):
        img_array = self.data_x[idx]
        label = self.data_y[idx]
        img_array = np.nan_to_num(img_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        img = Image.fromarray(img_array.astype(np.uint8)).convert('RGB')
        img = self.transform(img)
        
        if self.args.is_inpainting:
            if self.mode == "train":
                masked_img, masked_part, masked_position = self.apply_random_mask(img)
                return img, masked_img, masked_part, masked_position, label
            
            else: # self.mode == "test"
                if self.args.test_mask:
                    if self.args.test_mask_type == "random":
                        masked_img, masked_part, masked_position = self.apply_random_mask(img)
                    else: # mask_type == "center"
                        masked_img, masked_position = self.apply_center_mask(img)
                        
                return img, masked_img, masked_position, label
        else :
            if self.mode == "test" :
                if self.args.test_mask :
                    if self.args.test_mask_type == "random":
                        masked_img, _, _ = self.apply_random_mask(img)
                    else: # mask_type == "center"
                        masked_img, _ = self.apply_center_mask(img)
                    
                    img = masked_img
            return img, label
    
    def __len__(self):
        return len(self.data_x)
    
    def apply_random_mask(self, img, mask_h=None, mask_w=None):
        """Apply a random mask of mask_h x mask_w dimensions."""
        mask_h = mask_h if mask_h is not None else self.mask_h
        mask_w = mask_w if mask_w is not None else self.mask_w

        if self.img_h == mask_h:
            y1 = 0
        else:
            y1 = np.random.randint(0, self.img_h - mask_h)
    
        # Check if image width is the same as mask width
        if self.img_w == mask_w:
            x1 = 0
        else:
            x1 = np.random.randint(0, self.img_w - mask_w)
                    
        y2, x2 = y1 + mask_h, x1 + mask_w

        masked_part = img[:, y1:y2, x1:x2]
        masked_img = img.clone()
        masked_img[:, y1:y2, x1:x2] = 1

        return masked_img, masked_part, (y1, x1, y2, x2)

    def apply_center_mask(self, img, mask_h=None, mask_w=None):
        """Apply a center mask of mask_h x mask_w dimensions."""
        mask_h = mask_h if mask_h is not None else self.mask_h
        mask_w = mask_w if mask_w is not None else self.mask_w

        y1 = (self.img_h - mask_h) // 2
        x1 = (self.img_w - mask_w) // 2
        y2, x2 = y1 + mask_h, x1 + mask_w

        masked_img = img.clone()
        masked_img[:, y1:y2, x1:x2] = 1

        return masked_img, (y1, x1, y2, x2)

