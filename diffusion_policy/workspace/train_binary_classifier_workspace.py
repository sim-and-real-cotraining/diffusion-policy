import torch
from torch.utils.data import DataLoader, random_split, Subset
import wandb
import hydra
from tqdm import tqdm
from omegaconf import OmegaConf
from diffusion_policy.workspace.base_workspace import BaseWorkspace

class TrainBinaryClassifierWorkspace(BaseWorkspace):
    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        self.device = cfg.training.device

        # Dataloaders
        dataset = hydra.utils.instantiate(cfg.dataset)
        self.train_loader, self.val_loader = self.get_dataloaders(dataset)

        # Model, optimizer, loss
        self.model = hydra.utils.instantiate(cfg.model).to(self.device)
        self.optimizer = self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters()
        )
        pos_weight = dataset.get_num_zeros() / dataset.get_num_ones()
        self.loss_fn = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight]
        ).to(self.device))

    
    def run(self):
        wandb.init(project=self.cfg.name, config=self.cfg)

        for epoch in tqdm(range(self.cfg.training.num_epochs), desc="Epochs"):
            self.model.train()

            # Training
            epoch_train_loss = 0
            for inputs, labels in self.train_loader:
                # Forward pass
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels.unsqueeze(1).float())
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_train_loss += loss.item()

            # Validation loss and metrics
            self.model.eval()
            epoch_val_loss = 0
            validation_accuracy = 0
            total_val_datapoints = 0
            true_positives = 0
            false_positives = 0
            true_negatives = 0
            false_negatives = 0
            with torch.no_grad():
                for inputs, labels in self.val_loader:
                    # loss
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.loss_fn(outputs, labels.unsqueeze(1).float())
                    epoch_val_loss += loss.item()

                    # accuracy and confusion matrix
                    predictions = (torch.sigmoid(outputs) >= 0.5).float()
                    validation_accuracy += (predictions == labels.unsqueeze(1)).sum().item()
                    total_val_datapoints += len(labels)

                    true_positives += ((predictions == 1) & (labels.unsqueeze(1) == 1)).sum().item()
                    false_positives += ((predictions == 1) & (labels.unsqueeze(1) == 0)).sum().item()
                    true_negatives += ((predictions == 0) & (labels.unsqueeze(1) == 0)).sum().item()
                    false_negatives += ((predictions == 0) & (labels.unsqueeze(1) == 1)).sum().item()

            epoch_train_loss /= len(self.train_loader)
            epoch_val_loss /= len(self.val_loader)
            validation_accuracy /= total_val_datapoints

            # Calculate percentages for true/false positives/negatives
            predicted_positives = true_positives + false_positives
            predicted_negatives = true_negatives + false_negatives

            true_positive_percentage = true_positives / predicted_positives if predicted_positives > 0 else 0
            false_positive_percentage = false_positives / predicted_positives if predicted_positives > 0 else 0
            true_negative_percentage = true_negatives / predicted_negatives if predicted_negatives > 0 else 0
            false_negative_percentage = false_negatives / predicted_negatives if predicted_negatives > 0 else 0

            # Log metrics
            wandb.log(
                {
                    "train_loss": epoch_train_loss, 
                    "val_loss": epoch_val_loss,
                    "val_accuracy": validation_accuracy,
                    "true_positive_percentage": true_positive_percentage,
                    # "false_positive_percentage": false_positive_percentage,
                    "true_negative_percentage": true_negative_percentage,
                    # "false_negative_percentage": false_negative_percentage
                },
            )
        
        wandb.finish()

    def get_dataloaders(self, dataset):
        # Split dataset into training and validation
        val_size = int(len(dataset) * self.cfg.val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        # Balance the validation dataset
        val_indices_0 = [i for i in val_dataset.indices if dataset[i][1] == 0]
        val_indices_1 = [i for i in val_dataset.indices if dataset[i][1] == 1]
        balanced_val_size = min(len(val_indices_0), len(val_indices_1))
        balanced_val_indices = val_indices_0[:balanced_val_size] + val_indices_1[:balanced_val_size]
        balanced_val_dataset = Subset(dataset, balanced_val_indices)

        # Create data loaders
        train_loader = DataLoader(train_dataset, shuffle=True, **self.cfg.dataloader)
        val_loader = DataLoader(balanced_val_dataset, shuffle=False, **self.cfg.dataloader)

        return train_loader, val_loader


    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

    def predict(self, inputs):
        self.model.eval()
        if len(inputs.shape) == 1:
            inputs = inputs.reshape(1, -1)
        with torch.no_grad():
            inputs = torch.tensor(inputs, dtype=torch.float32).to(self.device)
            outputs = self.model(inputs)
            predicted_labels = (torch.sigmoid(outputs) >= 0.5)
            return outputs.cpu().numpy(), predicted_labels.cpu().numpy()

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)