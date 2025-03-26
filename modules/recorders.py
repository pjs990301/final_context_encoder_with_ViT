import os
from matplotlib import pyplot as plt
import pandas as pd
import logging
import torch
import csv
from sklearn.metrics import confusion_matrix
import numpy as np
from modules.utils import load_yaml
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

class Recorder():
    def __init__(self,
                name:str,
                record_dir:str,
                model: object,
                optimizer: object,
                scheduler: object,
                logger: logging.RootLogger=None):
        self.record_dir = record_dir
        self.plot_dir = os.path.join(record_dir, 'plots')
        self.record_filepath = os.path.join(self.record_dir, 'record.csv')
        self.weight_path = os.path.join(record_dir,'model', f'{name}_model.pt')
        
        self.logger = logger
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        os.makedirs(self.plot_dir, exist_ok = True)
        os.makedirs(os.path.join(record_dir,'model'), exist_ok = True)
        
    def set_model(self, model):
        self.model = model
        
    def set_logger(self, logger: logging.RootLogger):
        self.logger = logger
        
    def create_record_directory(self):
        os.makedirs(self.record_dir, exist_ok=True)
        
        msg = f"Create directory {self.record_dir}"
        self.logger.info(msg) if self.logger else None
        
    def add_row(self, row_dict: dict):
        fieldnames = list(row_dict.keys())
        
        with open(self.record_filepath, newline='', mode='a') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            if f.tell() == 0:
                writer.writeheader()
                
            writer.writerow(row_dict)
            msg = f"Write row {row_dict['epoch_index']}"
            self.logger.info(msg) if self.logger else None
            
    def save_weight(self, epoch:int) -> None:
        check_point = {
            'epoch': epoch + 1,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
        }
        torch.save(check_point, self.weight_path)
        msg = f"Recorder, epoch {epoch} Model saved: {self.weight_path}"
        self.logger.info(msg) if self.logger else None
        
    def save_plot(self, plots: list):
        record_df = pd.read_csv(self.record_filepath)
        current_epoch = record_df['epoch_index'].max()
        epoch_range = list(range(0, current_epoch+1))
        color_list = ['red', 'blue']  # train, val

        for plot_name in plots:
            # columns = [f'train_{plot_name}']
            columns = [f'train_{plot_name}', f'val_{plot_name}']

            fig = plt.figure(figsize=(20, 8))
            
            for id_, column in enumerate(columns):
                if column in record_df.columns:  # Check if the column exists in record_df

                    values = record_df[column].tolist()
                    plt.plot(epoch_range, values, marker='.', c=color_list[id_], label=column)
        
        # for plot_name in plots:
        #     columns = [f'train_{plot_name}']

        #     fig = plt.figure(figsize=(20, 8))
            
        #     for id_, column in enumerate(columns):
        #         values = record_df[column].tolist()
        #         plt.plot(epoch_range, values, marker='.', c=color_list[id_], label=column)
                 
            plt.title(plot_name, fontsize=15)
            # plt.legend(loc='upper right')
            # plt.grid()
            plt.xlabel('epoch')
            plt.ylabel(plot_name)
            plt.xticks(epoch_range, [str(i) for i in epoch_range])
            plt.gca().xaxis.set_major_locator(MultipleLocator(10))
            plt.tight_layout()
            plt.close(fig)
            fig.savefig(os.path.join(self.plot_dir, plot_name +'.png'))
        
    def save_confusion_matrix(self, y_true, y_pred, normalize=True, title=None, cmap=plt.cm.Blues, activity=None):
        xlabel_fontsize = 32
        ylabel_fontsize = 32
        xticks_fontsize = 26
        yticks_fontsize = 26

        mapping = load_yaml('config/mapping.yaml')
        
        
        # 기존 activity에 따른 방식
        if activity == "location" :
            classes = list(mapping["Location"].keys())
        elif activity == "action" :
            classes = list(mapping["Label"].keys())
        elif activity == "all" :
            classes = list(mapping["Location_Label"].keys())

        """
        이 함수는 혼동 행렬을 그리고, 선택적으로 정규화합니다.
        normalize=True`로 설정하면, 행렬의 요소를 클래스별 샘플 수로 나눕니다.
        """
        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'

        # 혼동 행렬 계산
        cm = confusion_matrix(y_true, y_pred)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
        # 혼동 행렬 저장하는 코드
        np.savetxt(os.path.join(self.plot_dir,'confusion_matrix.csv'), cm, delimiter=",", fmt="%.8f")

        fig, ax = plt.subplots(figsize=(15, 15))
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        # ax.figure.colorbar(im, ax=ax)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        ax.figure.colorbar(im, ax=ax, cax=cax)

        ax.set(xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=classes, yticklabels=classes,
           )

        # 라벨 회전
        ax.set_xlabel('Predicted label', fontsize=xlabel_fontsize)
        ax.set_ylabel('True label', fontsize=ylabel_fontsize)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=xticks_fontsize)
        plt.setp(ax.get_yticklabels(), fontsize=yticks_fontsize)

        # 각 셀에 숫자 표시
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir,'confusionMatrix.png'))
        plt.close(fig)
        save_avg_chart(cm, self.plot_dir)
        
def save_avg_chart(cm, plot_dir:str) :
    region_a = cm[1:4, 1:4]
    region_b = cm[4:7, 4:7]
    region_c = cm[7:10, 7:10]
    
    avg_region_a_correct = np.mean(np.diag(region_a))
    avg_region_b_correct = np.mean(np.diag(region_b))
    avg_region_c_correct = np.mean(np.diag(region_c))

    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    regions = ['A', 'B', 'C']
    averages_correct = [avg_region_a_correct, avg_region_b_correct, avg_region_c_correct]
    
    fig = plt.figure(figsize=(8, 6))
    
    plt.bar(regions, averages_correct, color=default_colors)
    plt.xlabel('Region')
    plt.ylabel('Average Value')
    plt.title('Average Accuracy by Region')
    plt.ylim(0, 1)  # 값의 범위가 0에서 1 사이임
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir,'avg_chart.png'))
    plt.close(fig)


    