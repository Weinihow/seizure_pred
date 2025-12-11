import torch
from torch import nn
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
import mne
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
import pytorch_lightning as pl
from sklearn.preprocessing import StandardScaler
import sys
import os
import time
# Add main directory to sys.path to import dataset
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from sklearn.metrics import f1_score
from main.metrics import basic_metric, log_results_to_md
from main.dataset import RawDataset

# GPU Setting
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# Config
config = {
    'dataset':{
        'train_ratio': 0.8,
        'sampling_rate': 256,
        'data_type': 'raw',
        'balance': False,
        'ictal_def': [15, 60]
    },
    'params':{
        'batch_size': 32,
        'class_weight': 0.9/0.1,
        'epoch': 80,
        'lr': 1e-4,
        'weight_decay': 1e-3,
        'alpha': 0.7,
        'gamma': 2,
        'threshold': 0.5
    },
    'eval':{
        'metrics': 'all',
        'save_metric': 'F1_score',
        'threshold_best': 0.5,
        'threshold_bestloss': 0.5,
        'threshold_last': 0.5
    }
}

pl.seed_everything(42, workers=True)

# Data Loading
def load_data():
    base_path = os.path.join(os.path.dirname(__file__), '..')
    # open json
    json_path = os.path.join(base_path, f"CHB_EEG/chb01/seizure_time_{config['dataset']['ictal_def'][0]}_{config['dataset']['ictal_def'][1]}.json")
    with open(json_path, 'r') as f:
        seizure_time = json.load(f)
    items = list(seizure_time.items())

    # devide datasets
    train_eeg_signal = []
    for (obj, info) in items[3:21]:
        file = os.path.join(base_path, f"CHB_EEG/chb01/{obj}.edf")
        data = mne.io.read_raw_edf(file, verbose=False)
        raw_data = data.get_data()
        # print(raw_data.shape)
        train_eeg_signal.append(raw_data)
    train_timepoints = items[3:21]

    val_eeg_signal = []
    for (obj, info) in items[:3]:
        file = os.path.join(base_path, f"CHB_EEG/chb01/{obj}.edf")
        data = mne.io.read_raw_edf(file, verbose=False)
        raw_data = data.get_data()
        # print(raw_data.shape)
        val_eeg_signal.append(raw_data)
    val_timepoints = items[:3]

    test_eeg_signal = []
    for (obj, info) in items[21:]:
        file = os.path.join(base_path, f"CHB_EEG/chb01/{obj}.edf")
        data = mne.io.read_raw_edf(file, verbose=False)
        raw_data = data.get_data()
        # print(raw_data.shape)
        test_eeg_signal.append(raw_data)
    test_timepoints = items[21:]

    train_dataset = RawDataset(train_eeg_signal, train_timepoints, mode='train', balance=config['dataset']['balance'])
    val_dataset = RawDataset(val_eeg_signal, val_timepoints, mode='val')
    test_dataset = RawDataset(test_eeg_signal, test_timepoints, mode='test')

    print("Before downsampling:")
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset

# Downsample negative samples to balance the dataset (1:1 ratio)
def downsample_negative_samples(dataset, random_seed=42):
    """
    Downsample negative samples to match the number of positive samples.
    Returns a new dataset with balanced classes.
    """
    np.random.seed(random_seed)

    # Get all labels
    labels = np.array([dataset[i][1] for i in range(len(dataset))])

    # Find indices of positive and negative samples
    pos_indices = np.where(labels == 1)[0]
    neg_indices = np.where(labels == 0)[0]

    print(f"Original - Positive: {len(pos_indices)}, Negative: {len(neg_indices)}, Ratio: {len(neg_indices)/len(pos_indices):.2f}:1")

    # If no positive samples, return original dataset
    if len(pos_indices) == 0:
        print("Warning: No positive samples found, returning original dataset")
        return dataset

    # Randomly sample negative indices to match positive count
    if len(neg_indices) > len(pos_indices):
        sampled_neg_indices = np.random.choice(neg_indices, size=len(pos_indices), replace=False)
    else:
        sampled_neg_indices = neg_indices

    # Combine positive and sampled negative indices
    balanced_indices = np.concatenate([pos_indices, sampled_neg_indices])
    np.random.shuffle(balanced_indices)  # Shuffle to mix classes

    # Create new balanced dataset
    balanced_data = [dataset.data[i] for i in balanced_indices]
    balanced_labels = [dataset.label[i] for i in balanced_indices]

    # Create a simple dataset-like object or update existing dataset
    class BalancedDataset:
        def __init__(self, data, labels):
            self.data = data
            self.label = labels

        def __len__(self):
            return len(self.label)

        def __getitem__(self, idx):
            return self.data[idx], self.label[idx]

    balanced_dataset = BalancedDataset(balanced_data, balanced_labels)

    # Verify balance
    balanced_labels_check = np.array([balanced_dataset[i][1] for i in range(len(balanced_dataset))])
    pos_count = np.sum(balanced_labels_check == 1)
    neg_count = np.sum(balanced_labels_check == 0)
    print(f"Balanced - Positive: {pos_count}, Negative: {neg_count}, Ratio: {neg_count/pos_count:.2f}:1")

    return balanced_dataset

# Model
class DCNN_BiLSTM(nn.Module):
    def __init__(self, input_channels=23, input_time=1280):
        super(DCNN_BiLSTM, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 2), stride=1, padding=(1, 0)),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(3, 2), stride=1, padding=(1, 0)),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(3, 2), stride=1, padding=(1, 0)),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(3, 2), stride=1, padding=(1, 0)),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.dropout2d = nn.Dropout2d(p=0.1)

        with torch.no_grad():
            dummy_x = torch.zeros(1, 1, input_channels, input_time)
            x = self.conv1(dummy_x)
            x = self.pool(x)
            x = self.conv2(x)
            x = self.pool(x)
            x = self.conv3(x)
            x = self.pool(x)
            x = self.conv4(x)
            self.lstm_input_size = x.shape[1] * x.shape[2] 
            self.lstm_seq_len = x.shape[3]
            print(f"CNN Output Shape: {x.shape}")
            print(f"LSTM Input Features: {self.lstm_input_size}, Seq Len: {self.lstm_seq_len}")

        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size, 
            hidden_size=20, 
            num_layers=1, 
            bidirectional=True, 
            batch_first=True
        )

        self.fc = nn.Sequential(
            nn.Linear(20 * 2, 1) # 40 -> 1
        )
        
        self.dropout = nn.Dropout(0.75)

    def forward(self, x):
        x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = self.conv4(x)

        x = x.permute(0, 3, 1, 2) # (Batch, W, 32, H)
        x = x.reshape(x.size(0), x.size(1), -1) # (Batch, W, 32*H)

        lstm_out, _ = self.lstm(x)

        x = lstm_out[:, -1, :] 
        
        x = self.dropout(x)

        x = self.fc(x)
        return x

def pick_best_threshold(y_true, y_prob):
    thresholds = np.linspace(0.05, 0.95, 19)
    scores = []
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        try:
            score = f1_score(y_true, preds)
        except Exception:
            score = 0.0
        scores.append((t, score))
    return max(scores, key=lambda x: x[1])

if __name__ == "__main__":
    # Prepare Data
    train_dataset, val_dataset, test_dataset = load_data()
    
    print("\nDownsampling training dataset...")
    train_dataset = downsample_negative_samples(train_dataset, random_seed=42)
    
    print(f"\nAfter downsampling:")
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    print("Normalizing data using Global StandardScaler...")
    # NOTE: Notebook logic equivalent
    # 1. Flatten all data to fit transform (Batch, Features)
    X_train = np.array([x.flatten() for x in train_dataset.data])
    X_val = np.array([x.flatten() for x in val_dataset.data])
    X_test = np.array([x.flatten() for x in test_dataset.data])

    # 2. Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 3. Reshape back to (Batch, 23, 1280) for the model
    # Original shape of one sample
    original_shape = train_dataset.data[0].shape
    train_dataset.data = [x.reshape(original_shape) for x in X_train_scaled]
    val_dataset.data = [x.reshape(original_shape) for x in X_val_scaled]
    test_dataset.data = [x.reshape(original_shape) for x in X_test_scaled]
    
    print("Global Normalization complete")

    # Dataloaders
    # Calculate class weights for sampling
    train_labels = np.array([train_dataset[i][1] for i in range(len(train_dataset))])
    class_counts = np.bincount(train_labels.astype(int))
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[train_labels.astype(int)]
    weighted_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['params']['batch_size'],
        sampler=weighted_sampler
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['params']['batch_size'],
        shuffle=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['params']['batch_size'],
        shuffle=False
    )

    # Model Setup
    input_shape = next(iter(train_loader))[0].shape
    model = DCNN_BiLSTM(input_channels=input_shape[1], input_time=input_shape[2]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['params']['lr'], weight_decay=config['params']['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    total_params = sum(p.numel() for p in model.parameters())
    print("Total parameters:", total_params)

    # Training Loop
    epochs = config['params']['epoch']
    best_score, best_epoch = 0, 0
    best_loss, best_loss_epoch = 1e9, 0
    train_losses, val_losses = [], []
    train_f1s, val_f1s = [], []
    
    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)
    
    # Model naming: DCNN_BiLSTM_1560_GlobalStd
    subject_id = "01" 
    ictal_start = config['dataset']['ictal_def'][0]
    ictal_end = config['dataset']['ictal_def'][1]
    model_base_name = f"01DCNN_BiLSTM_{ictal_start}{ictal_end}_Global"
    
    start_time = time.time()
    for epoch in range(epochs):
        print(f"Epoch: {epoch+1}--------------------------")
        model.train()
        train_loss = 0.0
        y_true_t, y_prob_t, y_pred_t = [], [], []
        for X, y in tqdm(train_loader):
            X = X.float().to(device)
            y = y.float().to(device)

            # Add Gaussian noise
            if model.training:
                noise = torch.randn_like(X) * 0.01
                X = X + noise

            logits = model(X).view(-1)
            loss = criterion(logits, y)
            train_loss += loss.item() * y.shape[0]
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            probs = torch.sigmoid(logits)
            y_true_t.extend(y.cpu().numpy())
            y_prob_t.extend(probs.detach().cpu().numpy().reshape(-1))
            y_pred_t.extend((probs >= config['params']['threshold']).long().cpu().numpy().reshape(-1))

        train_loss /= len(train_dataset)
        train_losses.append(train_loss)
        print(f"Train loss: {train_loss:.5f}")
        train_results = basic_metric(y_true_t, y_pred_t, y_prob_t, metrics=['f1_score', 'accuracy', 'fpr'])
        train_f1s.append(train_results['F1_score'])

        # Validation with best threshold
        model.eval()
        val_loss = 0.0
        y_true_v, y_prob_v = [], []
        with torch.no_grad():
            for X, y in tqdm(val_loader):
                X = X.float().to(device)
                y = y.float().to(device)
                logits = model(X).view(-1)
                loss = criterion(logits, y)
                val_loss += loss.item() * y.shape[0]
                probs = torch.sigmoid(logits)
                y_true_v.extend(y.cpu().numpy())
                y_prob_v.extend(probs.detach().cpu().numpy().reshape(-1))

        val_loss /= len(val_dataset)
        val_losses.append(val_loss)
        print(f"Val loss: {val_loss:.5f}")
        
        # Step the scheduler
        scheduler.step(val_loss)

        y_true_np = np.array(y_true_v)
        y_prob_np = np.array(y_prob_v)
        best_t, best_f1 = pick_best_threshold(y_true_np, y_prob_np)
        print(f"Val best threshold: {best_t:.2f}, F1: {best_f1:.4f}")
        y_pred_v = (y_prob_np >= best_t).astype(int).tolist()

        val_results = basic_metric(y_true_np, y_pred_v, y_prob_np, metrics=['all'])
        val_f1s.append(val_results['F1_score'])

        # Save by chosen metric and by loss
        if val_results[config['eval']['save_metric']] > best_score:
            best_score = val_results[config['eval']['save_metric']]
            torch.save(model.state_dict(), f'models/checkpoints/{model_base_name}_best.pth')
            print('model saved (best metric)')
            best_epoch = epoch + 1
            config['eval']['threshold_best'] = best_t
        if val_loss < best_loss:
            best_loss = val_loss
            best_loss_epoch = epoch + 1
            torch.save(model.state_dict(), f'models/{model_base_name}.pth')
            print('model saved (best loss)')
            config['eval']['threshold_bestloss'] = best_t

        # Always save last
        torch.save(model.state_dict(), f'models/checkpoints/{model_base_name}_last.pth')
        config['eval']['threshold_last'] = best_t
        print('model saved (last)')

    print(f'best epoch: {best_epoch}')
    print(f'best loss epoch: {best_loss_epoch}')
    print(f'best-F1 threshold: {config["eval"]["threshold_best"]}')
    print(f'best-loss threshold: {config["eval"]["threshold_bestloss"]}')

    # Plot Training Curve
    os.makedirs("result/training", exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss Curve during Training (Global)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'result/training/{model_base_name}_training_curve.png')
    print(f"Training curve saved to result/training/{model_base_name}_training_curve.png")
    
    # Plot F1 Curve
    plt.figure(figsize=(8, 4))
    plt.plot(train_f1s, label='Training F1')
    plt.plot(val_f1s, label='Validation F1')
    plt.title('F1 Curve during Training (Global)')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'result/training/{model_base_name}_f1_curve.png')
    print(f"F1 curve saved to result/training/{model_base_name}_f1_curve.png")
    

    
    end_time = time.time()
    training_duration = end_time - start_time

    # Test Section
    print("\n========== Test Section ==========")
    
    test_models = [
        ('Best Loss Model', f'models/{model_base_name}.pth', config['eval']['threshold_bestloss']),
        ('Best Metric Model', f'models/checkpoints/{model_base_name}_best.pth', config['eval']['threshold_best'])
    ]

    for name, path, threshold in test_models:
        if os.path.exists(path):
            print(f"\nTesting {name}...")
            model.load_state_dict(torch.load(path))
            model.eval()
            
            y_true_test, y_prob_test, y_pred_test = [], [], []
            with torch.no_grad():
                for X, y in tqdm(test_loader):
                    X = X.float().to(device)
                    y = y.float().to(device)
                    
                    logits = model(X).view(-1)
                    probs = torch.sigmoid(logits)
                    preds = (probs >= threshold).long()
                    
                    y_true_test.extend(y.cpu().numpy())
                    y_prob_test.extend(probs.cpu().numpy().reshape(-1))
                    y_pred_test.extend(preds.cpu().numpy().reshape(-1))
            
            print(f"Threshold: {threshold}")
            test_results = basic_metric(y_true_test, y_pred_test, y_prob_test, metrics=['all'])

            # Log to Markdown
            log_results_to_md(
                filename='result/training/training_log.md',
                model_name=model_base_name,
                model_type=name,
                threshold=threshold,
                results=test_results,
                training_duration=training_duration
            )
        else:
            print(f"\n{name} not found at {path}")
