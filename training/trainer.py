import os
import torch
from sklearn.metrics import f1_score
from tqdm import tqdm


class AverageMeter:
    def __init__(self):
        self.avg = 0.0
        self._count = 0

    def update(self, val):
        self._count += 1
        if self._count == 1:
            self.avg = float(val)
        else:
            self.avg = 0.2 * float(val) + 0.8 * self.avg


class Trainer:
    def __init__(self, model, loss_fn, optimizer, scheduler, device, is_mil, save_dir, use_wandb=False):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.is_mil = is_mil
        self.save_dir = save_dir
        self.use_wandb = use_wandb

    def _forward(self, x):
        out = self.model(x)
        return out[0] if isinstance(out, tuple) else out

    def train_epoch(self, loader):
        self.model.train()
        self.model.to(self.device)
        meter = AverageMeter()
        bar = tqdm(loader, ascii=True)
        for i, data in enumerate(bar):
            x, y = data[0].to(self.device), data[1].to(self.device)
            logits = self._forward(x)
            loss = self.loss_fn(logits.squeeze(1), y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            meter.update(loss.detach())
            if i % 5 == 0:
                bar.set_description(f'Loss {meter.avg:.4f}')
        return meter.avg

    def validate(self, loader, epoch):
        self.model.eval()
        self.model.to(self.device)
        loss_meter = AverageMeter()
        all_preds, all_gts = [], []

        with torch.no_grad():
            for data in tqdm(loader, ascii=True):
                x, y = data[0].to(self.device), data[1].to(self.device)
                logits = self._forward(x)
                loss = self.loss_fn(logits.squeeze(1), y)
                loss_meter.update(loss.detach())
                all_preds.extend((torch.sigmoid(logits.squeeze(1)) >= 0.5).int().cpu().numpy().tolist())
                all_gts.extend(y.int().cpu().numpy().tolist())

        val_f1 = f1_score(all_gts, all_preds, average='binary', zero_division=0)
        print(f'[Epoch {epoch+1}] Val F1: {val_f1:.6f}  Val Loss: {loss_meter.avg:.6f}')
        return val_f1, loss_meter.avg

    def fit(self, train_loader, val_loader, epochs):
        best_val_loss = float('inf')
        for epoch in range(epochs):
            print(f'Epoch {epoch+1}/{epochs}')
            train_loss = self.train_epoch(train_loader)
            val_f1, val_loss = self.validate(val_loader, epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                path = os.path.join(self.save_dir, f'best_val_epoch{epoch:03d}.pt')
                torch.save(self.model.state_dict(), path)

            if self.use_wandb:
                import wandb
                wandb.log({
                    'Train_Loss': train_loss,
                    'Val_Loss': val_loss,
                    'Val_F1': val_f1,
                    'lr': self.scheduler.optimizer.param_groups[0]['lr'],
                })

            self.scheduler.step()
