import os
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import timm
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.optim.swa_utils import AveragedModel, update_bn
from torch.amp import autocast, GradScaler
from timm.optim import AdamP
from timm.scheduler import CosineLRScheduler
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score, AUROC, AveragePrecision, ConfusionMatrix, ROC
from tqdm import tqdm

# Suppress Albumentations update warnings
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

# Configure memory allocation for CUDA
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('training.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# -------------------- Data Preprocessing --------------------
class AdvancedXRayDataset(Dataset):
    """Optimized X-ray dataset with Albumentations and dynamic sizing"""
    def __init__(self, file_paths, labels, img_size=224, is_training=False):
        self.file_paths = file_paths
        self.labels = labels
        self.img_size = img_size
        self.is_training = is_training

        # Validate labels
        assert all(label in [0, 1] for label in self.labels), "Labels must be 0 or 1"

        # Base transform (always applied)
        self.base_transform = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

        # Training augmentations (applied only if is_training=True)
        self.aug_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                     scale=(0.9, 1.1), rotate=(-15, 15), p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.CoarseDropout(num_holes_range=(8, 8), hole_height_range=(32, 32),
                            hole_width_range=(32, 32), p=0.3),
            A.GaussNoise(std_range=(0.012, 0.028), mean_range=(0, 0),
                         per_channel=True, noise_scale_factor=1.0, p=0.2),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3)
        ])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        try:
            image = np.array(Image.open(img_path).convert('RGB'))
            label = self.labels[idx]
            if self.is_training:
                image = self.aug_transform(image=image)['image']
            image = self.base_transform(image=image)['image']
            # Log image stats for first 5 images
            if idx < 5:
                logger.info(f"Image {idx}: min={image.min().item()}, max={image.max().item()}")
            return image, torch.tensor(label, dtype=torch.float32)
        except Exception as e:
            logger.warning(f"Error loading {img_path}: {e}")
            return torch.zeros(3, self.img_size, self.img_size), torch.tensor(0.0)

def create_stage_loaders(df, stage_config, train_paths, train_labels, val_paths, val_labels):
    """Create loaders for current training stage"""
    img_size = stage_config['img_size']
    batch_size = stage_config['batch_size']

    train_dataset = AdvancedXRayDataset(train_paths, train_labels, img_size=img_size, is_training=True)
    val_dataset = AdvancedXRayDataset(val_paths, val_labels, img_size=img_size)

    class_counts = np.bincount(train_labels.astype(int))
    class_weights = 1. / class_counts
    sample_weights = class_weights[train_labels.astype(int)]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    return {
        'train': DataLoader(train_dataset, batch_size, sampler=sampler,
                            num_workers=min(8, os.cpu_count()), pin_memory=True, persistent_workers=True),
        'val': DataLoader(val_dataset, batch_size, shuffle=False,
                          num_workers=min(4, os.cpu_count()), pin_memory=True)
    }

# -------------------- Model Architecture --------------------
class SimpleAttention(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_features, in_features // 16),
            nn.ReLU(),
            nn.Linear(in_features // 16, in_features),
            nn.Sigmoid()
        )

    def forward(self, x):
        attention_weights = self.attention(x)
        return x * attention_weights

class WeaponDetector(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.base_model = timm.create_model('tf_efficientnetv2_m', pretrained=True, features_only=False)
        in_features = self.base_model.classifier.in_features
        self.base_model.classifier = nn.Identity()
        self.attention = SimpleAttention(in_features)
        self.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        self.to(memory_format=torch.channels_last)

    def forward(self, x):
        x = x.contiguous(memory_format=torch.channels_last)
        features = self.base_model(x)
        features = self.attention(features)
        return self.head(features)

# -------------------- Training Utilities --------------------
class ProgressiveTrainer:
    def __init__(self, model, device, df, train_paths, train_labels, val_paths, val_labels):
        self.model = model
        self.device = device
        self.df = df
        self.train_paths = train_paths
        self.train_labels = train_labels
        self.val_paths = val_paths
        self.val_labels = val_labels
        self.stages = [
            {'img_size': 224, 'batch_size': 32, 'epochs': 10, 'lr': 1e-4},
            {'img_size': 300, 'batch_size': 16, 'epochs': 15, 'lr': 1e-4},
            {'img_size': 384, 'batch_size': 8, 'epochs': 20, 'lr': 5e-5}
        ]
        self.current_stage = 0
        self.global_epoch = 0
        self.total_epochs = sum(s['epochs'] for s in self.stages)

    def get_loaders(self):
        return create_stage_loaders(self.df, self.stages[self.current_stage],
                                    self.train_paths, self.train_labels,
                                    self.val_paths, self.val_labels)

    def progress_stage(self):
        if self.current_stage < len(self.stages) - 1:
            self.current_stage += 1
            return True
        return False

def tta_predict(model, image, device):
    preds = []
    with torch.no_grad():
        with torch.autocast(device_type="cuda"):
            output = model(image)
            preds.append(torch.sigmoid(output))
            flipped = transforms.functional.hflip(image)
            output = model(flipped)
            preds.append(torch.sigmoid(output))
            rotated = transforms.functional.rotate(image, angle=5)
            output = model(rotated)
            preds.append(torch.sigmoid(output))
    return torch.mean(torch.stack(preds), dim=0)

def train_model(csv_path):
    torch.cuda.empty_cache()

    df = pd.read_csv(csv_path)
    df = df[df['image_path'].apply(os.path.exists)].sample(frac=1, random_state=42)
    logger.info(f"Loaded dataset with {len(df)} samples")

    if df['label'].isnull().any() or not df['label'].isin([0, 1]).all():
        logger.error("Invalid labels in DataFrame")
        return None
    zeros, ones = len(df[df['label'] == 0]), len(df[df['label'] == 1])
    logger.info(f"Label distribution: 0s={zeros}, 1s={ones}, ratio={zeros/ones:.4f}")

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pos_weight = torch.tensor([zeros / ones]).to(device)
    logger.info(f"Using pos_weight: {pos_weight:.4f}")

    # Split data
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        df['image_path'].values, df['label'].values.astype('float32'),
        test_size=0.1, stratify=df['label'], random_state=42
    )
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels, test_size=0.1, stratify=train_val_labels, random_state=42
    )
    logger.info(f"Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")

    logger.info(f"Using device: {device}, CUDA: {torch.version.cuda}, PyTorch: {torch.__version__}")
    torch.backends.cudnn.benchmark = True

    # Model setup
    model = WeaponDetector().to(device)
    ema_model = AveragedModel(model).to(device)
    scaler = GradScaler("cuda")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Metrics
    metrics = MetricCollection({
        'accuracy': Accuracy(task='binary'), 'f1': F1Score(task='binary'),
        'auc': AUROC(task='binary'), 'prc': AveragePrecision(task='binary'),
        'confmat': ConfusionMatrix(task='binary'), 'roc': ROC(task='binary')
    }).to(device)

    # Initialize trainer
    trainer = ProgressiveTrainer(model, device, df, train_paths, train_labels, val_paths, val_labels)

    # Compile model if supported
    if hasattr(torch, 'compile'):
        model = torch.compile(model)
        ema_model = torch.compile(ema_model)

    best_score = 0.0
    patience = 5
    no_improve_epochs = 0
    best_auc = 0.0

    optimizer = AdamP(model.parameters(), lr=trainer.stages[0]['lr'], weight_decay=1e-4)
    scheduler = CosineLRScheduler(optimizer, t_initial=trainer.total_epochs, warmup_t=5, lr_min=1e-6, warmup_lr_init=1e-5)

    accumulation_steps = 4  # For gradient accumulation

    while True:
        stage_config = trainer.stages[trainer.current_stage]
        loaders = trainer.get_loaders()
        logger.info(f"\n=== Stage {trainer.current_stage + 1}: {stage_config} ===")

        for epoch in range(stage_config['epochs']):
            model.train()
            train_loss = []
            pbar = tqdm(loaders['train'], desc=f"Epoch {trainer.global_epoch + 1}")
            optimizer.zero_grad(set_to_none=True)

            for i, (images, labels) in enumerate(pbar):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.autocast(device_type="cuda"):
                    outputs = model(images).squeeze()
                    outputs = torch.clamp(outputs, min=-10, max=10)
                    loss = criterion(outputs, labels) / accumulation_steps

                scaler.scale(loss).backward()
                if (i + 1) % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    ema_model.update_parameters(model)

                train_loss.append(loss.item() * accumulation_steps)
                pbar.set_postfix({'loss': f"{np.mean(train_loss[-10:]):.4f}", 'lr': f"{optimizer.param_groups[0]['lr']:.2e}"})

            # Validation
            model.eval()
            metrics.reset()
            val_loss = []
            with torch.no_grad():
                for images, labels in loaders['val']:
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    with torch.autocast(device_type="cuda"):
                        outputs = ema_model(images).squeeze()
                        loss = criterion(outputs, labels)
                        preds = torch.sigmoid(outputs)
                        int_labels = labels.long()
                    val_loss.append(loss.item())
                    metrics.update(preds, int_labels)
            computed_metrics = metrics.compute()
            avg_metrics = {k: computed_metrics[k].item() for k in computed_metrics if k not in ['confmat', 'roc']}
            current_score = avg_metrics['auc'] + avg_metrics['prc']
            logger.info(f"Epoch {trainer.global_epoch + 1} - Train Loss: {np.mean(train_loss):.4f}, Val Loss: {np.mean(val_loss):.4f}")
            logger.info(f"Val Metrics: {avg_metrics}")
            logger.info(f"Confusion Matrix:\n{computed_metrics['confmat']}")

            # Early stopping
            if avg_metrics['auc'] > best_auc:
                best_auc = avg_metrics['auc']
                no_improve_epochs = 0
                if current_score > best_score:
                    best_score = current_score
                    torch.save({'model': model.state_dict(), 'ema_model': ema_model.state_dict(),
                                'stage': trainer.current_stage, 'epoch': trainer.global_epoch, 'metrics': avg_metrics},
                               'best_model.pth')
                    logger.info(f"New best model saved with score: {best_score:.4f}")
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= patience:
                    logger.info(f"Early stopping after {patience} epochs without AUC improvement")
                    break

            scheduler.step(trainer.global_epoch)
            trainer.global_epoch += 1

        update_bn(loaders['train'], ema_model, device=device)
        if not trainer.progress_stage():
            break

    # Test with TTA
    logger.info("\n=== Final Test Evaluation with TTA ===")
    test_dataset = AdvancedXRayDataset(test_paths, test_labels, img_size=384)
    test_loader = DataLoader(test_dataset, batch_size=8, num_workers=4, pin_memory=True)

    ema_model.eval()
    test_metrics = {k: [] for k in metrics.keys()}
    test_loss = []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            labels = labels.to(device)
            preds = tta_predict(ema_model, images, device)
            with torch.autocast(device_type="cuda"):
                outputs = ema_model(images).squeeze()
                loss = criterion(outputs, labels)
            int_labels = labels.long()
            test_loss.append(loss.item())
            batch_metrics = metrics(preds, int_labels)
            for k in test_metrics:
                test_metrics[k].append(batch_metrics[k].item())

    final_metrics = {k: np.mean(v) for k, v in test_metrics.items()}
    logger.info(f"Test Loss: {np.mean(test_loss):.4f}")
    logger.info(f"Final Test Metrics: {final_metrics}")

    return final_metrics

if __name__ == '__main__':
    DATA_CSV_PATH = 'baggage_labels.csv'
    if os.path.exists(DATA_CSV_PATH):
        logger.info("Starting training process...")
        final_metrics = train_model(DATA_CSV_PATH)
        if final_metrics is not None:
            logger.info("Training completed successfully!")
        else:
            logger.error("Training failed.")
    else:
        logger.error(f"Dataset file {DATA_CSV_PATH} not found!")