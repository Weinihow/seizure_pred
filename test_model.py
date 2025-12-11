import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import mne
import json
import sys
import os
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Add main directory to path to import dataset
sys.path.append('main')
from dataset import RawDataset

# Configuration (matching the training config)
config = {
    'dataset': {
        'sampling_rate': 256,
        'ictal_def': [15, 60],
        'balance': False 
    },
    'params': {
        'batch_size': 32,
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

# Model Definition (CNN_MLP)
class CNN_MLP(nn.Module):
    """
    CNN feature extractor followed by MLP classifier.
    """
    def __init__(self, input_channels=23, input_time=1280, dropout_p=0.3):
        super(CNN_MLP, self).__init__()

        # CNN Feature Extractor
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

        # Calculate flatten size
        dummy_x = torch.zeros(1, input_channels, input_time)
        dummy_x = dummy_x.unsqueeze(0) if dummy_x.ndim == 3 else dummy_x
        x = self.conv1(dummy_x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = self.conv4(x)
        flatten_size = x.numel()

        self.mlp = nn.Sequential(
            nn.Linear(flatten_size, 300),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(300, 100),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(50, 20),
            nn.Sigmoid(),
            nn.Dropout(0.3),
            nn.Linear(20, 1)
        )

    def forward(self, x):
        x = x.unsqueeze(1) # (B, 1, C, T)
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        return x

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

def load_data(patient_id):
    print(f"Loading data for patient {patient_id}...")
    
    # Load seizure times
    json_path = f"CHB_EEG/{patient_id}/seizure_time_{config['dataset']['ictal_def'][0]}_{config['dataset']['ictal_def'][1]}.json"
    if not os.path.exists(json_path):
        print(f"Error: JSON file not found at {json_path}")
        return [], []
    
    print(f"Reading labels from {json_path}")
    with open(json_path, 'r') as f:
        seizure_time = json.load(f)
    
    eeg_signal = []
    timepoints = []
    
    # Filter for the specific patient files
    edf_files = sorted(glob.glob(f"CHB_EEG/{patient_id}/*.edf"))
    
    for file_path in edf_files:
        filename = os.path.basename(file_path)
        file_key = filename.replace('.edf', '')
        
        if file_key in seizure_time:
            print(f"Processing {filename}...")
            try:
                data = mne.io.read_raw_edf(file_path, verbose=False)
                raw_data = data.get_data()
                # Ensure 23 channels
                if raw_data.shape[0] != 23:
                     # print(f"Warning: {filename} has {raw_data.shape[0]} channels, expected 23. Skipping or truncating.")
                     if raw_data.shape[0] > 23:
                         raw_data = raw_data[:23, :]
                     else:
                         continue
                
                eeg_signal.append(raw_data)
                timepoints.append((file_key, seizure_time[file_key]))
            except Exception as e:
                print(f"Error reading {filename}: {e}")

    return eeg_signal, timepoints

def test_model(model_path, patient_id='chb03', quiet=False):
    # 1. Load Data
    eeg_signal, timepoints = load_data(patient_id)
    
    if not eeg_signal:
        print("No data loaded.")
        return None

    # 2. Create Dataset
    if not quiet: print("Creating dataset...")
    test_dataset = RawDataset(eeg_signal, timepoints, mode='test', balance=False)
    
    if not quiet: print(f"Test dataset size: {len(test_dataset)}")
    
    # 3. Normalize
    if not quiet: print("Normalizing data...")
    test_dataset.data = normalize_per_sample(test_dataset.data)
    
    # 5. DataLoader
    test_loader = DataLoader(test_dataset, batch_size=config['params']['batch_size'], shuffle=False)
    
    # 6. Load Model
    if not quiet: print(f"Loading model from {model_path}...")
    
    # Try to detect model type
    try:
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint['state_dict'] if (isinstance(checkpoint, dict) and 'state_dict' in checkpoint) else checkpoint
        
        if any('lstm' in k for k in state_dict.keys()):
            if not quiet: print("Detected DCNN_BiLSTM model.")
            model = DCNN_BiLSTM().to(device)
        else:
            if not quiet: print("Detected CNN_MLP model.")
            model = CNN_MLP().to(device)
            
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None

    model.eval()
    
    # 7. Inference
    y_true = []
    y_pred = []
    y_prob = []
    
    if not quiet: print("Running inference...")
    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            inputs = batch_data.float().to(device)
            labels = batch_labels.float().to(device)
            
            outputs = model(inputs)
            
            # Apply sigmoid if not already applied (assuming logits output)
            probs = torch.sigmoid(outputs).squeeze()
            
            # Handle batch size 1 case
            if probs.ndim == 0:
                probs = probs.unsqueeze(0)
                
            preds = (probs > config['params']['threshold']).float()
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())
            
    # 8. Metrics
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = 0.0
        
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    results = {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1,
        "AUC": auc,
        "Specificity": specificity,
        "TN": tn, "FP": fp, "FN": fn, "TP": tp
    }

    if not quiet:
        print("\n" + "="*30)
        print(f"Results for {patient_id} using {os.path.basename(model_path)}")
        print("="*30)
        print(f"Accuracy:    {acc:.4f}")
        print(f"Precision:   {prec:.4f}")
        print(f"Recall:      {rec:.4f}")
        print(f"F1 Score:    {f1:.4f}")
        print(f"AUC:         {auc:.4f}")
        print(f"Specificity: {specificity:.4f}")
        print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        print("="*30)
        
    return results

if __name__ == "__main__":
    patients = ['chb01', 'chb03']
    models = ['best_model.pth', 'best_model_chb03.pth']
    
    # Verify files exist
    valid_models = [m for m in models if os.path.exists(m)]
    if not valid_models:
        print("No model files found.")
        sys.exit(1)
        
    print(f"Testing Models: {valid_models}")
    print(f"Testing Patients: {patients}")
    
    # Store results: results[patient][model] = metrics
    comparison = {}
    
    for patient in patients:
        comparison[patient] = {}
        print(f"\n{'='*50}")
        print(f"Processing Patient: {patient}")
        print(f"{'='*50}")
        
        for model_file in valid_models:
            print(f"  Testing with {model_file}...")
            res = test_model(model_file, patient_id=patient, quiet=True)
            if res:
                comparison[patient][os.path.basename(model_file)] = res
                print(f"    -> F1: {res['F1 Score']:.4f}, Acc: {res['Accuracy']:.4f}")
            else:
                print("    -> Failed")

    # Print Comparative Table
    print("\n" + "="*80)
    print("CROSS-PATIENT MODEL COMPARISON")
    print("="*80)
    
    # Header
    header = f"{'Patient':<10} | {'Metric':<12} | " + " | ".join([f"{m:<22}" for m in valid_models])
    print(header)
    print("-" * len(header))
    
    metrics_to_show = ["Accuracy", "F1 Score", "AUC", "Specificity"]
    
    for patient in patients:
        if patient not in comparison: continue
        
        first_metric = True
        for metric in metrics_to_show:
            row = f"{patient if first_metric else '':<10} | {metric:<12} | "
            for model_name in valid_models:
                if model_name in comparison[patient]:
                    val = comparison[patient][model_name][metric]
                    row += f"{val:<22.4f} | "
                else:
                    row += f"{'N/A':<22} | "
            print(row)
            first_metric = False
        print("-" * len(header))
