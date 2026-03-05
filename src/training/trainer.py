import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, List
import time


class Trainer:
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        
        self.model.to(self.device)
        
        self.train_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rates': []
        }
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        
        self.model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(images)
            
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            
            self.optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        return {
            'loss': epoch_loss,
            'accuracy': epoch_acc
        }
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        
        self.model.eval()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * images.size(0)
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        return {
            'loss': epoch_loss,
            'accuracy': epoch_acc
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            train_metrics = self.train_epoch(train_loader)
            
            val_metrics = self.validate(val_loader)
            
            self.train_history['train_loss'].append(train_metrics['loss'])
            self.train_history['val_loss'].append(val_metrics['loss'])
            self.train_history['train_acc'].append(train_metrics['accuracy'])
            self.train_history['val_acc'].append(val_metrics['accuracy'])
            
            if self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
            else:
                current_lr = self.optimizer.param_groups[0]['lr']
            
            self.train_history['learning_rates'].append(current_lr)
            
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
            
            epoch_time = time.time() - epoch_start_time
            
            if verbose:
                print(f"Epoch [{epoch+1}/{num_epochs}] - "
                      f"Time: {epoch_time:.2f}s - "
                      f"Train Loss: {train_metrics['loss']:.4f} - "
                      f"Train Acc: {train_metrics['accuracy']:.4f} - "
                      f"Val Loss: {val_metrics['loss']:.4f} - "
                      f"Val Acc: {val_metrics['accuracy']:.4f} - "
                      f"LR: {current_lr:.6f}")
        
        return self.train_history
    
    def get_history(self) -> Dict[str, List[float]]:
        return self.train_history
    
    def save_checkpoint(self, filepath: str, epoch: int, **kwargs):
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_history': self.train_history,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        checkpoint.update(kwargs)
        
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath: str) -> Dict:
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'train_history' in checkpoint:
            self.train_history = checkpoint['train_history']
        
        return checkpoint


if __name__ == "__main__":
    from src.models.model_factory import create_model
    from src.dataset.mri_dataset import MRISliceDataset, create_train_val_dataloaders
    from src.dataset.dataset_builder import build_dataset_from_volumes
    from src.dataset.split_utils import split_dataset_by_patient
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = create_model('cnn', num_classes=2, dropout_rate=0.5)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler
    )
    
    print("Trainer initialized successfully")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {device}")
