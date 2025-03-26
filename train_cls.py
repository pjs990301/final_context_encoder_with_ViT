import torch
from tqdm import tqdm

class CLS_Train():
    def __init__(self, model, optimizer, scheduler, loss, device, logger, metrics, interval=10):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss = loss
        self.device = device
        self.logger = logger
        self.interval = interval
        self.metrics = metrics
        
        # History
        self.loss_sum = 0 # Epoch loss sum
        self.loss_mean = 0 # epoch loss mean
        self.y = list()
        self.y_preds = list()
        self.score_dict = dict()
        self.acc_sum = 0
        self.acc_mean = 0
        
    def train(self, train_dataloader):
        
        self.model.train()
        train_loss, correct, total = 0.0, 0, 0
        
        for batch_index, (data_x, data_y) in enumerate(tqdm(train_dataloader)):

            x = data_x.to(self.device, dtype=torch.float)
            y = data_y.to(self.device, dtype=torch.long)
            
            y_pred = self.model(x)
            loss = self.loss(y_pred, y)
            
            self.optimizer.zero_grad()
            
            loss.backward()
            self.optimizer.step()
            
            _, predicted = torch.max(y_pred.data, 1)
            
            train_loss += loss.item() * x.size(0)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            
            # History
            self.loss_sum += loss.item()
            self.y_preds.extend(predicted.cpu().numpy())
            self.y.extend(y.cpu().numpy())
            
            if batch_index % self.interval == 0:
                msg = (f"[Batch {batch_index}/{len(train_dataloader)}] "
                       f"[Loss: {loss.item()}]")
                self.logger.info(msg)
                
        # epoch_loss = train_loss / len(train_dataloader.dataset)
        # epoch_acc = correct / total
        
        self.scheduler.step()
        
        # Epoch history
        self.loss_mean = self.loss_sum / len(train_dataloader.dataset)
        
        # Metric
        self.y_preds = torch.tensor(self.y_preds, dtype=torch.long)
        self.y = torch.tensor(self.y, dtype=torch.long)
        self.y_preds = self.y_preds.tolist()
        self.y = self.y.tolist()
        
        for metric_name, metric_func in self.metrics.items():
            score = metric_func(self.y, self.y_preds)
            self.score_dict[metric_name] = score
            
        # return epoch_loss, epoch_acc
        
    def clear_history(self):
        self.loss_sum = 0
        self.loss_mean = 0
        self.acc_mean = 0
        self.acc_sum = 0
        self.y_preds = list()
        self.y = list()
        self.score_dict = dict()