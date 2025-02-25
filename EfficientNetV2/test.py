import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import timm
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score, AUROC, AveragePrecision
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import logging
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('evaluation.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# -------------------- Dataset Definition --------------------
class AdvancedXRayDataset(Dataset):
    """Optimized X-ray dataset with Albumentations"""
    def __init__(self, file_paths, labels, img_size=384):
        self.file_paths = file_paths
        self.labels = labels
        self.img_size = img_size

        self.base_transform = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        try:
            image = np.array(Image.open(img_path).convert('RGB'))
            label = self.labels[idx]
            image = self.base_transform(image=image)['image']
            return image, torch.tensor(label, dtype=torch.float32)
        except Exception as e:
            logger.warning(f"Error loading {img_path}: {e}")
            return torch.zeros(3, self.img_size, self.img_size), torch.tensor(0.0)

# -------------------- Model Definition --------------------
class WeaponDetector(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.base_model = timm.create_model('tf_efficientnetv2_m', pretrained=False, features_only=False)
        in_features = self.base_model.classifier.in_features
        self.base_model.classifier = nn.Identity()
        
        self.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        features = self.base_model(x)
        return self.head(features)

# -------------------- Utility Functions --------------------
def load_data(csv_path):
    """Load and split data consistently with training"""
    df = pd.read_csv(csv_path)
    df = df[df['image_path'].apply(os.path.exists)]
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        df['image_path'].values, df['label'].values.astype('float32'),
        test_size=0.1, stratify=df['label'], random_state=42
    )
    return test_paths, test_labels

def create_test_loader(test_paths, test_labels, batch_size=32):
    """Create DataLoader for test set"""
    test_dataset = AdvancedXRayDataset(test_paths, test_labels, img_size=384)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return test_loader

def load_best_model(model, checkpoint_path='best_model.pth'):
    """Load the best model's EMA weights, adjusting for module prefix and EMA keys"""
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found at {checkpoint_path}")
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint['ema_model']
        
        # Remove 'module.' prefix if present (from DataParallel)
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("module.", "") if key.startswith("module.") else key
            if new_key != "n_averaged":  # Skip EMA-specific key
                new_state_dict[new_key] = value
        
        model.load_state_dict(new_state_dict)
        logger.info(f"Loaded best EMA model from {checkpoint_path} with adjusted state_dict")
        return model
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        raise

def evaluate_model(model, test_loader):
    """Evaluate the model on the test set"""
    model.eval()
    test_loss = []
    all_preds = []
    all_labels = []
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]).to(device))  # Match training

    metrics = MetricCollection({
        'accuracy': Accuracy(task='binary'),
        'precision': Precision(task='binary'),
        'recall': Recall(task='binary'),
        'f1': F1Score(task='binary'),
        'auc': AUROC(task='binary'),
        'prc': AveragePrecision(task='binary')
    }).to(device)

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating Test Set"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            preds = torch.sigmoid(outputs)

            test_loss.append(loss.item())
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    metric_values = metrics(all_preds, all_labels.long())
    avg_loss = np.mean(test_loss)

    return avg_loss, metric_values, all_labels, all_preds

# -------------------- Visualization Functions --------------------
def plot_confusion_matrix(labels, preds):
    labels = labels.numpy().astype(int)
    preds = preds.numpy()
    preds_binary = (preds > 0.5).astype(int)
    cm = confusion_matrix(labels, preds_binary)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()
    return specificity

def plot_roc_curve(labels, preds):
    labels = labels.numpy().astype(int)
    preds = preds.numpy()
    fpr, tpr, _ = roc_curve(labels, preds)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.close()

def plot_pr_curve(labels, preds):
    labels = labels.numpy().astype(int)
    preds = preds.numpy()
    precision, recall, _ = precision_recall_curve(labels, preds)
    
    plt.figure()
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig('pr_curve.png')
    plt.close()

def parse_training_logs(log_path='training.log'):
    train_losses = []
    val_losses = []
    epoch_pattern = re.compile(r"Epoch (\d+) Summary:")
    train_loss_pattern = re.compile(r"Train Loss: ([\d.]+)")
    val_loss_pattern = re.compile(r"Val Loss: ([\d.]+)")

    with open(log_path, 'r') as f:
        lines = f.readlines()
        current_epoch = 0
        for line in lines:
            epoch_match = epoch_pattern.search(line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
            train_match = train_loss_pattern.search(line)
            if train_match and current_epoch > 0:
                train_losses.append(float(train_match.group(1)))
            val_match = val_loss_pattern.search(line)
            if val_match and current_epoch > 0:
                val_losses.append(float(val_match.group(1)))

    min_len = min(len(train_losses), len(val_losses))
    return train_losses[:min_len], val_losses[:min_len]

def plot_loss_curves(train_losses, val_losses):
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Training Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_loss_curve.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, val_losses, label='Validation Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('validation_loss_curve.png')
    plt.close()

# -------------------- Main Function --------------------
def main():
    csv_path = 'baggage_labels.csv'
    checkpoint_path = 'best_model.pth'
    log_path = 'training.log'

    logger.info("Loading test data...")
    test_paths, test_labels = load_data(csv_path)
    test_loader = create_test_loader(test_paths, test_labels)

    logger.info("Parsing training logs for loss curves...")
    if os.path.exists(log_path):
        train_losses, val_losses = parse_training_logs(log_path)
        plot_loss_curves(train_losses, val_losses)
        logger.info("Loss curves saved: training_loss_curve.png, validation_loss_curve.png")
    else:
        logger.warning(f"Training log not found at {log_path}. Skipping loss plots.")

    logger.info("Initializing model...")
    model = WeaponDetector().to(device)
    model = load_best_model(model, checkpoint_path)

    logger.info("Evaluating model on test set...")
    test_loss, test_metrics, all_labels, all_preds = evaluate_model(model, test_loader)

    specificity = plot_confusion_matrix(all_labels, all_preds)
    plot_roc_curve(all_labels, all_preds)
    plot_pr_curve(all_labels, all_preds)

    print("\n=== Final Test Metrics ===")
    logger.info("\n=== Final Test Metrics ===")
    print(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test Loss: {test_loss:.4f}")
    for k, v in test_metrics.items():
        print(f"{k.capitalize()}: {v:.4f}")
        logger.info(f"{k.capitalize()}: {v:.4f}")
    print(f"Specificity: {specificity:.4f}")
    logger.info(f"Specificity: {specificity:.4f}")

    logger.info("Evaluation completed. Figures saved: confusion_matrix.png, roc_curve.png, pr_curve.png, training_loss_curve.png, validation_loss_curve.png")

if __name__ == '__main__':
    logger.info("Starting evaluation process...")
    main()