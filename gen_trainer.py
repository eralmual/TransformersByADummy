import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

class GenerativeTrainer:
    def __init__(self, model: nn.Module, train_loader: DataLoader, test_loader: DataLoader,
                 criterion: nn.Module, optimizer: optim.Optimizer, scheduler: optim.lr_scheduler = None,
                 device: str = 'cpu', model_name: str = "transformer", log_dir: str = 'runs') -> None:
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.writer = SummaryWriter(log_dir)
        self.model_name = model_name
        self.global_step = 0

        samples, y_train, _ = next(iter(self.test_loader))
        self.writer.add_graph(self.model, (samples.to(self.device), y_train.to(self.device)))

    def train_epoch(self, log_freq: int=500) -> float:
        # Prepare the model for training
        self.model.train()
        running_loss = 0.0
        # Run epoch
        for inputs, y_train, y_target in tqdm(self.train_loader, desc=f"Training", leave=False):
            # Move tensors to device
            inputs, y_train, y_target = inputs.to(self.device), y_train.to(self.device), y_target.to(self.device)
            
            # Zero out gradients and generate batch
            self.optimizer.zero_grad()
            outputs = self.model(inputs, y_train).contiguous().view(-1, self.model.out_tokenizer.get_vocab_size())
            # Calc loss without the start token and optimize
            loss = self.criterion(outputs, y_target.view(-1))
            loss.backward()
            self.optimizer.step()
            # Track loss
            running_loss += loss.item()
            self.global_step += 1
            
            if(self.global_step % log_freq == 0):
                self.writer.add_scalar('Train loss', loss.item(), self.global_step)
                running_loss = 0.0
            
        # Scheduler step
        if(self.scheduler is not None):
            self.scheduler.step()

        return running_loss / len(self.train_loader)

    def test(self) -> float:
        # Prepare the model for testing
        self.model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for inputs, y_train, y_target in tqdm(self.test_loader, desc="Testing", leave=False):
                # Move tensors to device
                inputs, y_train, y_target = inputs.to(self.device), y_train.to(self.device), y_target.to(self.device)

                # Test batch
                outputs = self.model(inputs, y_train).contiguous().view(-1, self.model.out_tokenizer.get_vocab_size())
                loss = self.criterion(outputs, y_target.view(-1))
                # Track loss
                running_loss += loss.item()
                
        return running_loss / len(self.test_loader)

    def train(self, num_epochs: int, test_interval: int = 2) -> None:
        # Track eval loss to save checkpoints
        best_loss = 10000
        for epoch in tqdm(range(num_epochs), desc="Epochs"):
            # Train the model
            train_loss = self.train_epoch()
            print(f'Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}')
            
            if ((epoch + 1) % test_interval == 0):
                # Track test loss and update
                test_loss = self.test()
                self.writer.add_scalar('Test loss', test_loss, epoch)
                print(f'Epoch {epoch+1}/{num_epochs}: Test Loss: {test_loss:.4f}')
                # Check if the model improved
                if(test_loss < best_loss):
                    best_loss = test_loss
                    torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    }, f"{self.model_name}.pth")
                    torch.save(self.model.state_dict(), 'best_model.pth')
                    print("Model saved!")


# Example usage:
# model = YourModel()
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
# trainer = GenerativeTrainer(model, train_loader, test_loader, criterion, optimizer, scheduler, device='cuda')
# trainer.train(num_epochs=10, test_interval=1)
