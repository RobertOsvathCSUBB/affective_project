import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import lightning.pytorch as L
from torchmetrics import Accuracy, ConfusionMatrix
from torch.utils.tensorboard import SummaryWriter

class DeepEmotion(nn.Module):
    def __init__(self):
        super(DeepEmotion,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(10, 10, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(10, 10, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(10, 10, kernel_size=3),
            nn.BatchNorm2d(10),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU()
        )

        self.dropout = nn.Dropout()

        self.fc = nn.Sequential(
            nn.Linear(10 * 9 * 9, 50),
            nn.Linear(50, 7)
        )

        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1, stride=2),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 10, kernel_size=3, padding=1, stride=2),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU()
        )

        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(),
            nn.Linear(32, 6)
        )
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 90)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x = F.grid_sample(x, grid, align_corners=True)

        return x

    def forward(self, input):
        out = self.stn(input)
        out = self.conv1(input)
        out = self.conv2(out)
        out = self.dropout(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return {
            'logits': out,
            'probs': F.softmax(out, dim=1)
        }
    

class EmotionClassifier(L.LightningModule):
    def __init__(self, lr, classes=7):
        super().__init__()
        self.save_hyperparameters()
        self.model = DeepEmotion()
        self.loss = nn.CrossEntropyLoss()
        self.acc = Accuracy(task='multiclass', num_classes=classes)
        self.test_acc = Accuracy(task='multiclass', num_classes=classes)
        self.confusion_matrix = ConfusionMatrix(task='multiclass', num_classes=classes)
        self.validation_step_outputs = []

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        output = self.model(imgs)
        loss = self.loss(output['logits'], labels)
        self.log('train_loss', loss.item(), on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        output = self.model(imgs)
        loss = self.loss(output['logits'], labels)
        self.log('val_loss', loss.item(), on_step=False, on_epoch=True, prog_bar=True)

        accuracy = self.acc(output["probs"], labels)
        self.log("val_acc", accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.validation_step_outputs.append({"val_loss": loss, "val_acc": accuracy})
        self.confusion_matrix(output["probs"], labels)
        return {"val_loss": loss, "val_acc": accuracy}

    def on_validation_epoch_end(self):
        avg_loss = torch.stack([x["val_loss"] for x in self.validation_step_outputs]).mean()
        avg_acc = torch.stack([x["val_acc"] for x in self.validation_step_outputs]).mean()

        self.log("val_loss", avg_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", avg_acc, on_step=False, on_epoch=True, prog_bar=True)

        self.validation_step_outputs.clear()

        cm = self.confusion_matrix.compute().cpu().numpy()
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        ax.set_title('Confusion Matrix')

        writer = SummaryWriter(log_dir=self.logger.log_dir)
        writer.add_figure('Confusion Matrix', fig, global_step=self.current_epoch)
        writer.close()

        self.confusion_matrix.reset()

    def predict_step(self, batch, batch_idx):
        events, _ = batch
        output = self.model(events)
        return torch.argmax(output["probs"], dim=1)

    def test_step(self, batch, batch_idx):
        imgs, labels = batch
        output = self.model(imgs)
        loss = self.loss(output['logits'], labels)
        self.log('test_loss', loss.item())

        accuracy = self.test_acc(output["probs"], labels)
        self.log("test_acc", accuracy)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = {
            'scheduler': optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.01, patience=5, verbose=True),
            'monitor': 'train_loss',
            'interval': 'epoch',
            'frequency': 1
        }
        return [optimizer], [scheduler]
    
    def forward(self, x):
        return self.model(x)
    
emotion_classifier = EmotionClassifier.load_from_checkpoint("fer_2.ckpt")