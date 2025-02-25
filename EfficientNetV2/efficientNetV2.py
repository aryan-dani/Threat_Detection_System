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
from torch.cuda.amp import GradScaler, autocast
from timm.optim import AdamP
from timm.scheduler import CosineLRScheduler
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score, AUROC, AveragePrecision
from tqdm import tqdm


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
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

        self.base_transform = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

         # Augmentation transform: applied only in training mode
        self.aug_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Affine(
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                scale=(0.9, 1.1),
                rotate=(-15, 15),
                p=0.5
            ),
            A.RandomBrightnessContrast(p=0.3),
            A.CoarseDropout(
                num_holes_range=(8, 8),
                hole_height_range=(32, 32),
                hole_width_range=(32, 32),
                p=0.3
            ),
            A.GaussNoise(
                std_range=(0.012, 0.028),
                mean_range=(0, 0),
                per_channel=True,
                noise_scale_factor=1.0,
                p=0.2
            ),
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
            return image, torch.tensor(label, dtype=torch.float32)
        except Exception as e:
            logger.warning(f"Error loading {img_path}: {e}")
            return torch.zeros(3, self.img_size, self.img_size), torch.tensor(0.0)


def create_stage_loaders(df, stage_config, train_paths, train_labels, val_paths, val_labels):
    """Create loaders for current training stage"""
    img_size = stage_config['img_size']
    batch_size = stage_config['batch_size']

    # Training dataset with augmentations
    train_dataset = AdvancedXRayDataset(train_paths, train_labels, 
                                      img_size=img_size, is_training=True)
    
    # Validation dataset
    val_dataset = AdvancedXRayDataset(val_paths, val_labels, img_size=img_size)

    # Class-balanced sampler
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
class WeaponDetector(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.base_model = timm.create_model('tf_efficientnetv2_m', pretrained=True, features_only=False)
        in_features = self.base_model.classifier.in_features
        self.base_model.classifier = nn.Identity()
        
        self.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
        # Use channels_last memory format for better performance
        self.to(memory_format=torch.channels_last)
    def forward(self, x):
        x = x.contiguous(memory_format=torch.channels_last)
        features = self.base_model(x)
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
            {'img_size': 224, 'batch_size': 64, 'epochs': 10, 'lr': 3e-4},
            {'img_size': 300, 'batch_size': 32, 'epochs': 15, 'lr': 1e-4},
            {'img_size': 384, 'batch_size': 16, 'epochs': 20, 'lr': 5e-5}
        ]
        self.current_stage = 0
        self.global_epoch = 0
        self.total_epochs = sum(s['epochs'] for s in self.stages)

    def get_loaders(self):
        return create_stage_loaders(self.df, 
                                   self.stages[self.current_stage],
                                   self.train_paths,
                                   self.train_labels,
                                   self.val_paths,
                                   self.val_labels)

    def progress_stage(self):
        if self.current_stage < len(self.stages) - 1:
            self.current_stage += 1
            return True
        return False


def train_model(csv_path):
    # Data preparation
    df = pd.read_csv(csv_path)
    df = df[df['image_path'].apply(os.path.exists)].sample(frac=1, random_state=42)
    logger.info(f"Loaded dataset with {len(df)} samples")
    
    # Split data
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        df['image_path'].values, df['label'].values.astype('float32'),
        test_size=0.1, stratify=df['label'], random_state=42
    )
    
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels,
        test_size=0.1, stratify=train_val_labels, random_state=42
    )
    logger.info(f"Train samples: {len(train_paths)}, Val samples: {len(val_paths)}, Test samples: {len(test_paths)}")
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"CUDA version: {torch.version.cuda}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"Using device: {device}")
    torch.backends.cudnn.benchmark = True

    # Model setup
    model = WeaponDetector().to(device)
    ema_model = AveragedModel(model).to(device)
    scaler = torch.amp.GradScaler("cuda")
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]).to(device))  # Adjust based on your data

    # Metrics
    metrics = MetricCollection({
        'accuracy': Accuracy(task='binary'),
        'f1': F1Score(task='binary'),
        'auc': AUROC(task='binary'),
        'prc': AveragePrecision(task='binary')
    }).to(device)

    # Initialize trainer
    trainer = ProgressiveTrainer(model, device, df, train_paths, train_labels, val_paths, val_labels)
    best_score = 0.0

    optimizer = AdamP(model.parameters(), lr=trainer.stages[0]['lr'], weight_decay=1e-4)
    scheduler = CosineLRScheduler(optimizer,
                                t_initial=trainer.total_epochs,
                                warmup_t=5,
                                lr_min=1e-6,
                                warmup_lr_init=1e-5)
    # Training loop
    while True:
        stage_config = trainer.stages[trainer.current_stage]
        loaders = trainer.get_loaders()
        logger.info(f"\n=== Stage {trainer.current_stage + 1}: {stage_config} ===")
        logger.info(f"Stage configuration: {stage_config}")
        logger.info(f"Current learning rate: {optimizer.param_groups[0]['lr']:.2e}")


        for epoch in range(stage_config['epochs']):
            # Training phase
            model.train()
            train_loss = []
            pbar = tqdm(loaders['train'], desc=f"Epoch {trainer.global_epoch + 1}")
            for images, labels in pbar:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.amp.autocast("cuda"):
                    outputs = model(images).squeeze()
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                ema_model.update_parameters(model)

                train_loss.append(loss.item())
                pbar.set_postfix({
                    'loss': f"{np.mean(train_loss[-10:]):.4f}",
                    'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
                })

            # Validation phase
            model.eval()
            val_loss = []
            val_metrics = {k: [] for k in metrics.keys()}
            with torch.no_grad():
                for images, labels in loaders['val']:
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)

                    with torch.amp.autocast("cuda"):

                        outputs = ema_model(images).squeeze()
                        loss = criterion(outputs, labels)
                        preds = torch.sigmoid(outputs.squeeze())
                        int_labels = labels.long()  # Convert labels to long tensor
                    
                    val_loss.append(loss.item())
                    batch_metrics = metrics(preds, int_labels)
                    for k in val_metrics:
                        val_metrics[k].append(batch_metrics[k].item())

            avg_metrics = {k: np.mean(v) for k, v in val_metrics.items()}
            current_score = avg_metrics['auc'] + avg_metrics['prc']
            logger.info(f"\nEpoch {trainer.global_epoch + 1} Summary:")
            logger.info(f"Train Loss: {np.mean(train_loss):.4f}")
            logger.info(f"Val Loss: {np.mean(val_loss):.4f}")
            logger.info(f"Val Metrics:")
            for k, v in avg_metrics.items():
                logger.info(f"- {k}: {v:.4f}")

            # Save best model
            if current_score > best_score:
                best_score = current_score
                torch.save({
                    'model': model.state_dict(),
                    'ema_model': ema_model.state_dict(),
                    'stage': trainer.current_stage,
                    'epoch': trainer.global_epoch,
                    'metrics': avg_metrics
                }, 'best_model.pth')
                logger.info(f"New best model saved with score: {best_score:.4f}")

            scheduler.step(trainer.global_epoch)
            trainer.global_epoch += 1

        # Update BN statistics
        logger.info("Updating batch normalization statistics...")
        update_bn(loaders['train'], ema_model, device=device)
        
        if not trainer.progress_stage():
            logger.info("Final training stage completed")
            break

    # Final evaluation on test set
    logger.info("\n=== Final Test Evaluation ===")
    test_dataset = AdvancedXRayDataset(test_paths, test_labels, img_size=384)
    test_loader = DataLoader(test_dataset, 32, num_workers=4, pin_memory=True)
    
    ema_model.eval()
    test_metrics = {k: [] for k in metrics.keys()}
    test_loss = []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            labels = labels.to(device)

            with torch.amp.autocast(device_type="cuda"):

                outputs = ema_model(images).squeeze()
                loss = criterion(outputs, labels)
                preds = torch.sigmoid(outputs)
                int_labels = labels.long()

            test_loss.append(loss.item())
            batch_metrics = metrics(preds, labels)
            for k in test_metrics:
                test_metrics[k].append(batch_metrics[k].item())

    final_metrics = {k: np.mean(v) for k, v in test_metrics.items()}
    logger.info("\n=== Final Test Metrics ===")
    logger.info(f"Test Loss: {np.mean(test_loss):.4f}")
    for k, v in final_metrics.items():
        logger.info(f"{k}: {v:.4f}")


if __name__ == '__main__':
    DATA_CSV_PATH = 'baggage_labels.csv'
    logger.info("Starting training process...")
    final_metrics = train_model(DATA_CSV_PATH)
    logger.info("Training completed successfully!")