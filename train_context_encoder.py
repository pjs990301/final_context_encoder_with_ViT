import torch
from tqdm import tqdm

class Inpainting_Train():
    def __init__(self, model, optimizer, scheduler, loss, device, logger, metrics, patch, interval=10):
        self.CLS = model[0]
        self.G = model[1]
        self.D = model[2]
        
        self.optimizer_CLS = optimizer[0]
        self.optimizer_G = optimizer[1]
        self.optimizer_D = optimizer[2]

        self.scheduler_CLS = scheduler[0]
                
        self.loss_CLS = loss[0]
        self.loss_adversarial = loss[1]
        self.loss_pixelwise = loss[2]
        
        self.device_cls = device[0]
        self.device_G = device[1]
        
        self.logger = logger
        self.interval = interval
        self.metrics = metrics
        self.patch = patch
        
        # History
        self.adv_loss_sum = 0
        self.adv_loss_mean = 0
        self.pixel_loss_sum = 0
        self.pixel_loss_mean = 0
        self.loss_sum = 0
        self.loss_mean = 0
        self.y = list()
        self.y_preds = list()
        self.score_dict = dict()
        self.acc_sum = 0
        self.acc_mean = 0
        
    def train(self, train_dataloader, epoch):
        self.CLS.train()
        self.G.train()
        self.D.train()
        
        for batch_index, (img, masked_img, masked_parts, mask_coordinates, label) in enumerate(tqdm(train_dataloader)):
            valid = torch.ones((img.shape[0], *self.patch), device=self.device_G, requires_grad=False)
            fake = torch.zeros((img.shape[0], *self.patch), device=self.device_G, requires_grad=False)
            
            img, masked_img, masked_parts = img.to(self.device_G), masked_img.to(self.device_G), masked_parts.to(self.device_G)
            
            # -----------------
            #  Train Generator
            # -----------------
            self.optimizer_G.zero_grad()
            gen_parts = self.G(masked_img)
            g_adv = self.loss_adversarial(self.D(gen_parts), valid)
            g_pixel = self.loss_pixelwise(gen_parts, masked_parts)
            g_loss = 0.001 * g_adv + 0.999 * g_pixel
            g_loss.backward()
            self.optimizer_G.step()
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            self.optimizer_D.zero_grad()
            real_loss = self.loss_adversarial(self.D(masked_parts), valid)
            fake_loss = self.loss_adversarial(self.D(gen_parts.detach()), fake)
            d_loss = 0.5 * (real_loss + fake_loss)
            d_loss.backward()
            self.optimizer_D.step()
            
            # -----------------
            #  Train Classifier
            # -----------------
            self.optimizer_CLS.zero_grad()
            filled_samples = masked_img.clone()
            y1, x1, y2, x2 = mask_coordinates[0][0].item(), mask_coordinates[1][0].item(), mask_coordinates[2][0].item(), mask_coordinates[3][0].item()
            filled_samples[:, :, y1:y2, x1:x2] = gen_parts.detach()
            vit_input = filled_samples.to(self.device_cls)
            
            y = label.to(self.device_cls, dtype=torch.long)
            y_pred = self.CLS(vit_input)
            cls_loss = self.loss_CLS(y_pred, y)
            cls_loss.backward()
            self.optimizer_CLS.step()
            
            self.adv_loss_sum += g_adv.item()
            self.pixel_loss_sum += g_pixel.item()
            self.loss_sum += cls_loss.item()
            
            _, predicted = torch.max(y_pred.data, 1)
            
            self.y.extend(y.cpu().numpy())
            self.y_preds.extend(predicted.cpu().numpy())

            batches_done = epoch * len(train_dataloader) + batch_index
            
            if batch_index % self.interval == 0:
                msg = (f"[Batch {batch_index}/{len(train_dataloader)}] "
                       f"[G Loss: {g_loss.item()}] "
                       f"[D Loss: {d_loss.item()}] "
                       f"[CLS Loss: {cls_loss.item()}]")
                self.logger.info(msg)
                
        self.scheduler_CLS.step()
        
        # Epoch history
        self.adv_loss_mean = self.adv_loss_sum / len(train_dataloader.dataset)
        self.pixel_loss_mean = self.pixel_loss_sum / len(train_dataloader.dataset)
        self.loss_mean = self.loss_sum / len(train_dataloader.dataset)
        
        # Metric
        self.y_preds = torch.tensor(self.y_preds, dtype=torch.long)
        self.y = torch.tensor(self.y, dtype=torch.long)
        self.y_preds = self.y_preds.tolist()
        self.y = self.y.tolist()
        
        for metric_name, metric_func in self.metrics.items():
            score = metric_func(self.y, self.y_preds)
            self.score_dict[metric_name] = score
            
    
    def clear_history(self):
        self.adv_loss_sum = 0
        self.adv_loss_mean = 0
        self.pixel_loss_sum = 0
        self.pixel_loss_mean = 0
        self.loss_sum = 0
        self.loss_mean = 0
        self.y = list()
        self.y_preds = list()
        self.score_dict = dict()
        self.acc_sum = 0
        self.acc_mean = 0
    