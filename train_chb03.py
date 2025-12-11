import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import mne
import json
import sys
import os
import glob
import copy
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Add main directory to path to import dataset
sys.path.append('main')
from dataset import RawDataset

# Configuration
config = {
    'dataset': {
        'sampling_rate': 256,
        'ictal_def': [15, 60],
        'balance': False 
    },
    'params': {
        'batch_size': 32,
        'epoch': 80,
        'lr': 1e-4,
        'weight_decay': 1e-3,
        'threshold': 0.5
    }
}

# Device configuration
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# Set seed
torch.manual_seed(42)
np.random.seed(42)

# Model Definition (DCNN_BiLSTM)
class DCNN_BiLSTM(nn.Module):
    def __init__(self, input_channels=23, input_time=1280):
        super(DCNN_BiLSTM, self).__init__()

        # 1. DCNN Front-end
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

        # 2. Calculate CNN output dimensions
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

        # 3. Bi-LSTM Back-end
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size, 
            hidden_size=20, 
            num_layers=1, 
            bidirectional=True, 
            batch_first=True
        )

        # 4. Classifier
        self.fc = nn.Sequential(
            nn.Linear(20 * 2, 1) 
        )
        
        self.dropout = nn.Dropout(0.75)

    def forward(self, x):
        # 1. Dimension adjustment: (B, C, T) -> (B, 1, C, T)
        x = x.unsqueeze(1)

        # 2. DCNN Feature Extraction
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = self.conv4(x)

        # 3. Reshape for LSTM
        x = x.permute(0, 3, 1, 2) # (Batch, W, 32, H)
        x = x.reshape(x.size(0), x.size(1), -1) # (Batch, W, 32*H)

        # 4. Bi-LSTM Processing
        lstm_out, _ = self.lstm(x)

        # 5. Take last time step
        x = lstm_out[:, -1, :] 
        
        x = self.dropout(x)

        # 6. Classification
        x = self.fc(x)
        return x

def normalize_per_sample(data_list):
    """Normalize each sample independently"""
    normalized = []
    for sample in data_list:
        sample_flat = sample.flatten()
        mean = np.mean(sample_flat)
        std = np.std(sample_flat)
        if std > 1e-8:
            normalized.append((sample - mean) / std)
        else:
            normalized.append(sample)
    return normalized

def downsample_negative_samples(dataset, random_seed=42):
    """
    Downsample negative samples to match the number of positive samples.
    Returns a new dataset with balanced classes.
    """
    np.random.seed(random_seed)

    # Get all labels
    labels = np.array([dataset.label[i] for i in range(len(dataset))])

    # Find indices of positive and negative samples
    pos_indices = np.where(labels == 1)[0]
    neg_indices = np.where(labels == 0)[0]

    print(f"Original - Positive: {len(pos_indices)}, Negative: {len(neg_indices)}, Ratio: {len(neg_indices)/len(pos_indices) if len(pos_indices)>0 else 0:.2f}:1")

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
    # We can reuse RawDataset structure but just update data/label
    new_dataset = copy.deepcopy(dataset)
    new_dataset.data = balanced_data
    new_dataset.label = balanced_labels
    
    # Verify balance
    balanced_labels_check = np.array(new_dataset.label)
    pos_count = np.sum(balanced_labels_check == 1)
    neg_count = np.sum(balanced_labels_check == 0)
    print(f"Balanced - Positive: {pos_count}, Negative: {neg_count}, Ratio: {neg_count/pos_count if pos_count>0 else 0:.2f}:1")

    return new_dataset

def load_and_split_data(patient_id='chb03'):
    print(f"Loading data for patient {patient_id}...")
    
    # Load seizure times
    json_path = f"CHB_EEG/{patient_id}/seizure_time_{config['dataset']['ictal_def'][0]}_{config['dataset']['ictal_def'][1]}.json"
    
    print(f"Reading labels from {json_path}")
    with open(json_path, 'r') as f:
        seizure_time = json.load(f)
    
    items = list(seizure_time.items())
    print(f"Total files found: {len(items)}")
    
    # Split: Val (3), Train (14), Test (3) - Total 20
    # Adjust indices based on total files
    # chb03 has 20 files.
    # Val: items[:3]
    # Train: items[3:17]
    # Test: items[17:]
    
    val_items = items[:3]
    train_items = items[3:17]
    test_items = items[17:]
    
    print(f"Split: Val={len(val_items)}, Train={len(train_items)}, Test={len(test_items)}")
    
    def load_files(file_items):
        eeg_signal = []
        timepoints = []
        for (file_key, info) in file_items:
            file_path = f"CHB_EEG/{patient_id}/{file_key}.edf"
            print(f"Processing {file_key}...")
            try:
                data = mne.io.read_raw_edf(file_path, verbose=False)
                raw_data = data.get_data()
                # Ensure 23 channels
                if raw_data.shape[0] != 23:
                     if raw_data.shape[0] > 23:
                         raw_data = raw_data[:23, :]
                     else:
                         continue
                eeg_signal.append(raw_data)
                timepoints.append((file_key, info))
            except Exception as e:
                print(f"Error reading {file_key}: {e}")
        return eeg_signal, timepoints

    print("\nLoading Validation Data...")
    val_eeg, val_time = load_files(val_items)
    print("\nLoading Training Data...")
    train_eeg, train_time = load_files(train_items)
    print("\nLoading Test Data...")
    test_eeg, test_time = load_files(test_items)
    
    return (train_eeg, train_time), (val_eeg, val_time), (test_eeg, test_time)

def train():
    # 1. Load Data
    (train_eeg, train_time), (val_eeg, val_time), (test_eeg, test_time) = load_and_split_data()
    
    # 2. Create Datasets
    print("\nCreating datasets...")
    train_dataset = RawDataset(train_eeg, train_time, mode='train', balance=False)
    val_dataset = RawDataset(val_eeg, val_time, mode='val')
    test_dataset = RawDataset(test_eeg, test_time, mode='test')
    
    print(f"Before downsampling: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    # 3. Downsample Training Data
    print("\nDownsampling training dataset...")
    train_dataset = downsample_negative_samples(train_dataset)
    
    # 4. Normalize
    print("\nNormalizing data...")
    train_dataset.data = normalize_per_sample(train_dataset.data)
    val_dataset.data = normalize_per_sample(val_dataset.data)
    test_dataset.data = normalize_per_sample(test_dataset.data)
    
    # 5. DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=config['params']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['params']['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['params']['batch_size'], shuffle=False)
    
    # 6. Model Setup
    model = DCNN_BiLSTM().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['params']['lr'], weight_decay=config['params']['weight_decay'])
    
    best_f1 = 0.0
    best_loss = float('inf')
    
    # History tracking
    history = {
        'train_loss': [], 'val_loss': [],
        'train_f1': [], 'val_f1': []
    }
    
    print("\nStarting training...")
    for epoch in range(config['params']['epoch']):
        model.train()
        train_loss = 0
        train_preds = []
        train_targets = []
        
        for batch_data, batch_labels in train_loader:
            inputs = batch_data.float().to(device)
            labels = batch_labels.float().to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            probs = torch.sigmoid(outputs).squeeze()
            preds = (probs > config['params']['threshold']).float()
            train_preds.extend(preds.cpu().detach().numpy())
            train_targets.extend(labels.cpu().detach().numpy())
            
        train_loss /= len(train_loader)
        train_acc = accuracy_score(train_targets, train_preds)
        train_f1 = f1_score(train_targets, train_preds, zero_division=0)
        
        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                inputs = batch_data.float().to(device)
                labels = batch_labels.float().to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels)
                val_loss += loss.item()
                
                probs = torch.sigmoid(outputs).squeeze()
                preds = (probs > config['params']['threshold']).float()
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_f1 = f1_score(val_targets, val_preds, zero_division=0)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)
        
        print(f"Epoch {epoch+1}/{config['params']['epoch']} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} F1: {train_f1:.4f} | Val Loss: {val_loss:.4f} F1: {val_f1:.4f}")
        
        # Save Best Models
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({'state_dict': model.state_dict()}, "best_model_chb03.pth")
            print("  Saved best_model_chb03.pth (Best F1)")
            
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({'state_dict': model.state_dict()}, "best_loss_model_chb03.pth")
            print("  Saved best_loss_model_chb03.pth (Best Loss)")

    # Plotting
    print("\nGenerating training plots...")
    plt.figure(figsize=(12, 5))
    
    # Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # F1 Curve
    plt.subplot(1, 2, 2)
    plt.plot(history['train_f1'], label='Training F1')
    plt.plot(history['val_f1'], label='Validation F1')
    plt.title('F1 Score Curve')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves_chb03.png')
    print("Saved plot to training_curves_chb03.png")

    # Final Test Evaluation
    print("\n" + "="*30)
    print("FINAL TEST EVALUATION (Test Split)")
    print("="*30)
    
    # Load best model
    print("Loading best_model_chb03.pth for testing...")
    checkpoint = torch.load("best_model_chb03.pth", map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    test_preds = []
    test_targets = []
    test_probs = []
    
    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            inputs = batch_data.float().to(device)
            labels = batch_labels.float().to(device)
            
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).squeeze()
            
            if probs.ndim == 0:
                probs = probs.unsqueeze(0)
                
            preds = (probs > config['params']['threshold']).float()
            
            test_preds.extend(preds.cpu().numpy())
            test_targets.extend(labels.cpu().numpy())
            test_probs.extend(probs.cpu().numpy())
            
    acc = accuracy_score(test_targets, test_preds)
    f1 = f1_score(test_targets, test_preds, zero_division=0)
    try:
        auc = roc_auc_score(test_targets, test_probs)
    except:
        auc = 0.0
    
    tn, fp, fn, tp = confusion_matrix(test_targets, test_preds).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"Test Accuracy:    {acc:.4f}")
    print(f"Test F1 Score:    {f1:.4f}")
    print(f"Test AUC:         {auc:.4f}")
    print(f"Test Specificity: {specificity:.4f}")
    print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print("="*30)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_only', action='store_true', help='Run only the test evaluation using existing model')
    args = parser.parse_args()
    
    if args.test_only:
        # Load data but only use test set
        print("Running in TEST ONLY mode...")
        (_, _), (_, _), (test_eeg, test_time) = load_and_split_data()
        test_dataset = RawDataset(test_eeg, test_time, mode='test')
        print("Normalizing test data...")
        test_dataset.data = normalize_per_sample(test_dataset.data)
        test_loader = DataLoader(test_dataset, batch_size=config['params']['batch_size'], shuffle=False)
        
        model = DCNN_BiLSTM().to(device)
        print("Loading best_model_chb03.pth...")
        if not os.path.exists("best_model_chb03.pth"):
            print("Error: best_model_chb03.pth not found. Run training first.")
            sys.exit(1)
            
        checkpoint = torch.load("best_model_chb03.pth", map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        
        test_preds = []
        test_targets = []
        test_probs = []
        
        with torch.no_grad():
            for batch_data, batch_labels in test_loader:
                inputs = batch_data.float().to(device)
                labels = batch_labels.float().to(device)
                outputs = model(inputs)
                probs = torch.sigmoid(outputs).squeeze()
                if probs.ndim == 0: probs = probs.unsqueeze(0)
                preds = (probs > config['params']['threshold']).float()
                test_preds.extend(preds.cpu().numpy())
                test_targets.extend(labels.cpu().numpy())
                test_probs.extend(probs.cpu().numpy())
        
        acc = accuracy_score(test_targets, test_preds)
        f1 = f1_score(test_targets, test_preds, zero_division=0)
        try:
            auc = roc_auc_score(test_targets, test_probs)
        except:
            auc = 0.0
        tn, fp, fn, tp = confusion_matrix(test_targets, test_preds).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        print("\n" + "="*30)
        print("TEST SET EVALUATION (Files 36-38)")
        print("="*30)
        print(f"Test Accuracy:    {acc:.4f}")
        print(f"Test F1 Score:    {f1:.4f}")
        print(f"Test AUC:         {auc:.4f}")
        print(f"Test Specificity: {specificity:.4f}")
        print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        print("="*30)
        
    else:
        train()
