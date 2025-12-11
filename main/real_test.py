import torch
from torch import nn
from torch.utils.data import DataLoader
import mne
import numpy as np

import matplotlib.pyplot as plt
import os
import re
import sys
import json
from tqdm import tqdm

# Add main directory to sys.path to import dataset
sys.path.append(os.path.dirname(__file__))

from dataset import RealTestDataset
from metrics import basic_metric

# GPU Setting
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# ===========================
# Model Definitions
# ===========================

class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(input_size, 300),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(300, 100),
            nn.ReLU(),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
        )
        self.fc4 = nn.Sequential(
            nn.Linear(50, 20),
            nn.Sigmoid(),
        )
        self.fc5 = nn.Sequential(
            nn.Linear(20, 1)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        return x

class CNN_MLP(nn.Module):
    def __init__(self, input_channels=23, input_time=1280, dropout_p=0.3):
        super(CNN_MLP, self).__init__()
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

        # Dummy forward to calculate flat size
        with torch.no_grad():
            dummy_x = torch.zeros(1, 1, input_channels, input_time)
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
        x = x.unsqueeze(1)
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

        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=20,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

        self.fc = nn.Sequential(
            nn.Linear(20 * 2, 1)
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
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(x.size(0), x.size(1), -1)
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]
        x = self.dropout(x)
        x = self.fc(x)
        return x

# ===========================
# Helper Functions
# ===========================

def normalize_per_sample_per_channel(data_list):
    """
    Normalize each channel per sample (per window):
    For each window: For each channel, subtract mean and divide by std.
    """
    normalized = []
    for sample in data_list:
        mean = np.mean(sample, axis=1, keepdims=True)
        std = np.std(sample, axis=1, keepdims=True) + 1e-8
        normalized.append((sample - mean) / std)
    return normalized

def load_realtest_segments(idx_list, base_path, dataset_name):
    """Load EEG signals for the given index list."""
    signals = []
    for idx in idx_list:
        file = f"{base_path}/{dataset_name}_{idx:02d}.edf"
        if not os.path.exists(file):
             print(f"Warning: File not found {file}")
             continue
        raw = mne.io.read_raw_edf(file, verbose=False)
        data = raw.get_data()
        print(f"Loaded {os.path.basename(file)} shape: {data.shape}")
        signals.append(data)
    return signals

def init_model(model_type, input_channels, input_time):
    if 'mlp' in model_type.lower() and 'cnn' not in model_type.lower():
         return MLP(input_size=input_channels*input_time).to(device)
    elif 'cnn' in model_type.lower() and 'lstm' not in model_type.lower():
         return CNN_MLP(input_channels=input_channels, input_time=input_time).to(device)
    elif 'lstm' in model_type.lower():
         return DCNN_BiLSTM(input_channels=input_channels, input_time=input_time).to(device)
    else:
         raise ValueError(f"Unknown model type: {model_type}")

def get_real_test_indices(dataset_name):
    if dataset_name == 'chb01':
        return [3, 4, 6] # 03/04 has seizure, 06 only interictal
    elif dataset_name == 'chb03':
        return list(range(1, 11)) + [15] + [24] + list(range(31, 39))
    else:
        # Default fallback or TODO
        return []

def get_seizure_time_json(dataset_name, ictal_def):
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    return os.path.join(base_path, f"CHB_EEG/{dataset_name}/seizure_time_{ictal_def[0]}_{ictal_def[1]}.json")

def plot_and_save(dataset_name, idx, raw_data, prediction, save_path, seizure_info):
    # Upsample prediction to match raw data length
    prediction = np.repeat(prediction, 5 * 256) # 5 seconds per prediction, fs=256
    # Truncate if mismatch
    min_len = min(len(prediction), raw_data.shape[1])
    prediction = prediction[:min_len]
    raw_channel = raw_data[1, :min_len]
    
    x1 = np.arange(min_len) / 256
    x2 = np.arange(min_len) # prediction is now same valid length
    
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(x1, raw_channel, label=f"EEG", linewidth=0.5, alpha=0.7)
    ax1.set_xlabel("Time (sec)")
    ax1.set_ylabel("EEG amplitude")
    ax1.set_title(f"Real Test - {dataset_name}_{idx:02d}")

    ax2 = ax1.twinx()
    # Align prediction with time
    x2_sec = np.arange(len(prediction)) / 256
    ax2.plot(x2_sec, prediction, color="red", label="Prediction", linewidth=1.5)
    ax2.set_ylabel("Prediction Probability (Not Thresholded)")
    # If prediction is binary, use step, if prob, use plot. Assuming input is binary preds for now based on original code
    
    y1_min, y1_max = ax1.get_ylim()
    ymax = max(np.abs(y1_max), np.abs(y1_min))
    ax1.set_ylim(-ymax, ymax)
    ax2.set_ylim(-0.1, 1.1)

    # Plot seizure timing lines
    if seizure_info:
        for key in ['interictal_start_time', 'interictal_end_time', 
                    'preictal_start_time', 'preictal_end_time', 'seizure_end_time']:
            if key in seizure_info and seizure_info[key]:
                color = 'green' if 'interictal' in key else ('blue' if 'preictal' in key else 'purple')
                ax1.axvline(x=seizure_info[key], color=color, linestyle="--", label=key.replace('_time', ''))

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    # Remove duplicates from legend
    unique_labels = []
    unique_lines = []
    for l, line in zip(labels, lines):
        if l not in unique_labels:
            unique_labels.append(l)
            unique_lines.append(line)
            
    plt.legend(unique_lines, unique_labels, loc="upper right")
    plt.savefig(save_path)
    plt.close()

# ===========================
# Main Process
# ===========================
if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    models_dir = os.path.join(base_dir, 'models')
    result_dir = os.path.join(base_dir, 'result', 'real_test')
    os.makedirs(result_dir, exist_ok=True)

    # Get all .pth files
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
    if not model_files:
        print("No models found in models/")
        sys.exit()

    print(f"Found {len(model_files)} models.")

    for model_file in model_files:
        print(f"\nProcessing model: {model_file}")
        
        # Parse model file name
        # Expected format: {Subject}{Model}_{IctalDef}_{Comment}.pth
        # Example: 01DCNN_BiLSTM_1560_best.pth
        
        # Regex to capture Subject (digits), ModelType (letters/underscores), Optional IctalDef
        # Try strict pattern first: 01DCNN_BiLSTM_1560.pth
        match = re.match(r"(\d+)([a-zA-Z_]+)_(\d+)(.*)\.pth", model_file)
        
        if match:
            subject_id = match.group(1)
            model_type_str = match.group(2)
            ictal_def_str = match.group(3)
        else:
            # Try simpler pattern: 01DCNN.pth
            match_simple = re.match(r"(\d+)([a-zA-Z_]+)\.pth", model_file)
            if match_simple:
                subject_id = match_simple.group(1)
                model_type_str = match_simple.group(2)
                ictal_def_str = "1560" # Default
                print(f"  > Filename {model_file} has no ictal def, defaulting to 1560")
            else:
                print(f"Skipping file {model_file} (naming pattern mismatch)")
                continue
        dataset_name = f"chb{subject_id}"
        
        # Ictal Def Parsing
        if len(ictal_def_str) == 4:
            ictal_def = [int(ictal_def_str[:2]), int(ictal_def_str[2:])]
        else:
            ictal_def = [15, 60] # default
            
        # Load Seizure Info
        json_path = get_seizure_time_json(dataset_name, ictal_def)
        if not os.path.exists(json_path):
            print(f"JSON not found: {json_path}")
            seizure_time = {}
        else:
            with open(json_path, 'r') as f:
                seizure_time = json.load(f)

        # Load Real Test Data
        idx_list = get_real_test_indices(dataset_name)
        if not idx_list:
            print(f"No real test indices defined for {dataset_name}")
            continue

        # Filter out already processed files
        base_filename = os.path.basename(model_file).replace('.pth', '')
        indices_to_process = []
        for idx in idx_list:
            save_name = f"{base_filename}_Test_{dataset_name}_{idx:02d}.png"
            save_path = os.path.join(result_dir, save_name)
            if os.path.exists(save_path):
                print(f"Skipping {save_name} (already exists)")
            else:
                indices_to_process.append(idx)
        
        idx_list = indices_to_process
        
        if not idx_list:
            print(f"All files for {model_file} already processed.")
            continue

        eeg_dir = os.path.join(base_dir, 'CHB_EEG', dataset_name)
        
        # Load Model
        # Need input dims from data first? No, fixed 23, 1280 usually.
        # But let's load logic first to be safe about dims if possible.
        # Actually default is usually 23, 1280.
        
        # Load data first to verify input shape
        # But loading all data for index list might be heavy, do per file?
        # Original code loads all.
        
        raw_signals = load_realtest_segments(idx_list, eeg_dir, dataset_name)
        if not raw_signals:
            continue
            
        test_dataset = RealTestDataset(raw_signals)
        # Normalize PER SAMPLE PER CHANNEL as per original code
        test_dataset.data = normalize_per_sample_per_channel(test_dataset.data)
        
        data_loader = DataLoader(
            test_dataset,
            batch_size=32, # Config batch size
            shuffle=False
        )
        
        sample_batch = next(iter(data_loader))
        # shape: (B, C, T)
        input_channels = sample_batch.shape[1]
        input_time = sample_batch.shape[2]
        
        model = init_model(model_type_str, input_channels, input_time)
        
        # Load weights
        checkpoint = torch.load(os.path.join(models_dir, model_file), map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint) # raw state dict
            
        threshold = checkpoint.get('threshold', 0.5) if isinstance(checkpoint, dict) else 0.5
        print(f"Loaded threshold: {threshold}")
        
        model.eval()
        all_preds = []
        with torch.no_grad():
            for X in tqdm(data_loader, desc="Inference"):
                X = X.float().to(device)
                logits = model(X).view(-1)
                probs = torch.sigmoid(logits)
                preds = (probs >= threshold).long() # Binary predictions
                all_preds.extend(preds.cpu().numpy())
        
        all_preds = np.array(all_preds)
        
        # Now visualize
        # Prediction is on 5-sec windows.
        # Need to map back to original files.
        # RealTestDataset flattens all files.
        # We need to know boundaries.
        
        current_pred_idx = 0
        for i, idx in enumerate(idx_list):
            raw_data = raw_signals[i]
            n_windows = raw_data.shape[1] // (5 * 256) # 5 sec windows
            
            file_preds = all_preds[current_pred_idx : current_pred_idx + n_windows]
            current_pred_idx += n_windows
            
            # Save plot
            base_filename = os.path.basename(model_file).replace('.pth', '')
            save_name = f"{base_filename}_Test_{dataset_name}_{idx:02d}.png"
            save_path = os.path.join(result_dir, save_name)
            
            file_key = f"{dataset_name}_{idx:02d}"
            info = seizure_time.get(file_key, {})
            
            plot_and_save(dataset_name, idx, raw_data, file_preds, save_path, info)
            print(f"Saved plot to {save_path}")
            
            # Helper for metric calculation
            def count_alarms_in_window(preds, start_idx, end_idx):
                if start_idx >= end_idx: return 0
                # Ensure within bounds
                s = max(0, start_idx)
                e = min(len(preds), end_idx)
                if s >= e: return 0
                return np.sum(preds[s:e])

            # Calculate Metrics for Logging
            # info keys: 'preictal_end_time' -> Seizure Start
            #            'seizure_end_time' -> Seizure End
            
            seizure_start = info.get('preictal_end_time')
            seizure_end = info.get('seizure_end_time')
            
            # Defaults
            valid_alarms = 0
            early_alarms = 0
            missed = "N/A"
            post_alarms = 0
            last_post_time_str = "-"
            fpr_h = 0.0
            
            POINTS_PER_SEC = 256 # Data
            SEC_PER_PRED = 5 # Prediction resolution
            
            if len(file_preds) > 0:
                total_duration_sec = len(file_preds) * SEC_PER_PRED
                
                if seizure_start is not None and seizure_end is not None:
                    # Seizure File
                    # Indices
                    sz_start_idx = int(seizure_start / SEC_PER_PRED)
                    sz_end_idx = int(seizure_end / SEC_PER_PRED)
                    
                    # Valid: 15 min before seizure
                    valid_start_idx = int((seizure_start - 15*60) / SEC_PER_PRED)
                    valid_alarms = np.sum(file_preds[max(0, valid_start_idx):sz_start_idx])
                    
                    # Early: 60 min to 15 min before
                    early_start_idx = int((seizure_start - 60*60) / SEC_PER_PRED)
                    early_alarms = np.sum(file_preds[max(0, early_start_idx):valid_start_idx])
                    
                    # Missed?
                    missed = "Yes" if (valid_alarms + early_alarms) == 0 else "No"
                    
                    # Post Seizure
                    post_alarms_arr = file_preds[sz_end_idx:]
                    post_alarms = np.sum(post_alarms_arr)
                    
                    if post_alarms > 0:
                        # Find last alarm index relative to post_alarms_arr
                        last_idx = np.where(post_alarms_arr == 1)[0][-1]
                        last_time_sec = (last_idx + 1) * SEC_PER_PRED
                        m, s = divmod(last_time_sec, 60)
                        last_post_time_str = f"+{int(m)}m {int(s)}s"
                        
                    # FPR (Interictal): Exclude [Preictal (1h?) + Seizure + Postictal?]
                    # Simply: Total alarms - (Valid + Early + Post) ? 
                    # Or defined as Alarms in "Normal" period.
                    # Typically Interictal is > 4h from seizure. But for this short file:
                    # Let's count "Other Alarms" = Total - (Valid + Early + Post + Ictal)
                    
                    # Ictal Alarms
                    ictal_alarms = np.sum(file_preds[sz_start_idx:sz_end_idx])
                    
                    false_alarms = np.sum(file_preds) - (valid_alarms + early_alarms + ictal_alarms + post_alarms)
                    # Duration for FPR: Total - (1h preictal + Seizure duration)
                    # Approx duration
                    fpr_duration_h = (total_duration_sec - (60*60) - (seizure_end - seizure_start)) / 3600
                    if fpr_duration_h > 0:
                        fpr_h = false_alarms / fpr_duration_h
                    
                else:
                    # Interictal File
                    missed = "-"
                    # All alarms are False Alarms
                    false_alarms = np.sum(file_preds)
                    fpr_h = false_alarms / (total_duration_sec / 3600)
                
                # Log to file
                log_file = os.path.join(base_dir, 'result', 'real_test', 'test_log.md')
                header = "| Model Name | File | Valid Warnings (<15m) | Early Warnings (15-60m) | Missed | Post-Seizure Alarms | Last Post-Alarm Time | False Alarms | FPR/h |\n|---|---|---|---|---|---|---|---|---|\n"
                
                if not os.path.exists(log_file):
                    with open(log_file, 'w') as f:
                        f.write(header)
                
                row = f"| {model_file} | {dataset_name}_{idx:02d} | {valid_alarms} | {early_alarms} | {missed} | {post_alarms} | {last_post_time_str} | {int(false_alarms)} | {fpr_h:.2f} |\n"
                with open(log_file, 'a') as f:
                    f.write(row)
                print(f"Logged metrics to {log_file}")

    print("\nAll models processed.")

