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

# Manual Threshold Override
# Format: 'Model_Base_Name': Threshold_Value
MANUAL_THRESHOLDS = {
    # Example:
    # '01DCNN_BiLSTM_1515_ratio10.pth': 0.25,
}

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
        return [3, 4, 5, 6] # Targeted files
    elif dataset_name == 'chb03':
        return list(range(1, 11)) + [15] + [24] + list(range(31, 39))
    else:
        # Default fallback or TODO
        return []

def load_threshold_from_log(model_name_base, target_metric_type='Best Metric Model'):
    """
    Parse result/training/training_log.md to find the best threshold for the model.
    Looking for row matching model_name_base and target_metric_type.
    target_metric_type example: 'Best Metric Model' or 'Best Loss Model'
    """
    log_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'result', 'training', 'training_log.md'))
    if not os.path.exists(log_path):
        return None
    
    best_threshold = None
    try:
        with open(log_path, 'r') as f:
            lines = f.readlines()
            # Table Header: | Date | Model Name | Model Type | Threshold | ...
            for line in lines:
                if '|' not in line: continue
                parts = [p.strip() for p in line.split('|')]
                if len(parts) < 5: continue
                
                # parts[0] is empty, parts[1] is Date, parts[2] is Model Name, parts[3] is Model Type
                row_model_name = parts[2]
                row_model_type = parts[3]
                row_threshold = parts[4]
                
                if row_model_name == model_name_base and target_metric_type in row_model_type:
                    try:
                        best_threshold = float(row_threshold)
                    except ValueError:
                        pass
                    # Keep looking for latest entry
    except Exception as e:
        print(f"Error reading log: {e}")
        
    return best_threshold

def get_seizure_time_json(dataset_name, ictal_def):
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    return os.path.join(base_path, f"CHB_EEG/{dataset_name}/seizure_time_{ictal_def[0]}_{ictal_def[1]}.json")

def plot_and_save(dataset_name, idx, raw_data, prediction, save_path, seizure_info, sampling_rate=256, mode='bg'):
    '''
    mode: bg: background / l: line / all: both
    '''
    
    # Upsample prediction to match raw data length
    prediction = np.repeat(prediction, 5) # 5 seconds per prediction, fs=256
    # # Truncate if mismatch
    # min_len = min(len(prediction), raw_data.shape[1])
    # prediction = prediction[:min_len]
    raw_channel = raw_data[0]
    
    x1 = np.arange(len(raw_channel)) / sampling_rate
    x2 = np.arange(len(prediction)) # prediction is now same valid length
    
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(x1, raw_channel, label=f"EEG", linewidth=0.5, alpha=0.7)
    ax1.set_xlabel("Time (sec)")
    ax1.set_ylabel("EEG amplitude")
    ax1.set_title(f"Real Test - {dataset_name}_{idx:02d}")

    ax2 = ax1.twinx()
    # Align prediction with time
    if mode == 'l' or mode == 'all':
        ax2.plot(x2, prediction, color="red", label="Prediction", linewidth=1.5)
        ax2.set_ylabel("Prediction Probability (Not Thresholded)")
    # If prediction is binary, use step, if prob, use plot. Assuming input is binary preds for now based on original code
    if mode == 'bg' or mode == 'all':
        for i, _ in enumerate(prediction):
            if prediction[i] == 1:
                plt.axvspan(i, i+1, fc = 'red', alpha = 0.3)
    
    y1_min, y1_max = ax1.get_ylim()
    ymax = max(np.abs(y1_max), np.abs(y1_min))
    ax1.set_ylim(-ymax, ymax)
    ax2.set_ylim(-1.1, 1.1)

    # Plot seizure timing lines
    if seizure_info:
        for key in ['interictal_start_time', 'interictal_end_time', 'interictal_start_time_2', 'interictal_end_time_2',
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

    # Define test targets
    test_targets = []
    
    # Standardized: All models in models/ are now deemed 'Best Metric Model' variants
    # or at least the primary models we want to test.
    # We treat them all as candidates for 'Best Metric Model' threshold lookup first.
    
    model_files = sorted([f for f in os.listdir(models_dir) if f.endswith('.pth')])
    
    if not model_files:
        print("No models found in models/.")
        sys.exit()

    print(f"Found {len(model_files)} models in models/")

    for f in model_files:
        test_targets.append({
            'path': os.path.join(models_dir, f),
            'filename': f,
            'base_name': f.replace('.pth', ''),
            'type': 'Best Metric Model' # Default assumption for primary models now
        })

    for target in test_targets:
        model_file = target['filename']
        model_path = target['path']
        model_base_name = target['base_name']
        model_type_label = target['type']
        
        print(f"\nProcessing: {model_file} ({model_type_label})")
        
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
            match_simple = re.match(r"(\d+)([a-zA-Z_]+)\.pth", model_file.replace('_best.pth', '.pth')) # Handle _best suffix for regex
            if match_simple:
                subject_id = match_simple.group(1)
                model_type_str = match_simple.group(2)
                ictal_def_str = "1560" # Default
                print(f"  > Filename {model_file} has no ictal def, defaulting to 1560")
            else:
                print(f"Skipping file {model_file} (naming pattern mismatch)")
                continue
        
        # Check if already tested
        log_file = os.path.join(base_dir, 'result', 'real_test', 'test_log.md')
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                log_content = f.read()
            if f"| {model_file} |" in log_content:
                print(f"Skipping {model_file} (Already in test_log.md)")
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

        # Filter out already processed files based on indices
        base_filename = os.path.basename(model_file).replace('.pth', '')
        
        # For aggregation, we want to run ALL indices to get a correct full statistic.
        # But if charts exist, user might want to skip.
        # However, to regenerate the log line correctly, we really should re-run inference on all.
        # Or blindly trust existing PNGs? No, logic requested is to check chb01_03,04,05,06.
        # The user requested to delete old results anyway.
        
        eeg_dir = os.path.join(base_dir, 'CHB_EEG', dataset_name)
        
        # Load data
        # Check files existence first
        idx_list = [i for i in idx_list if os.path.exists(os.path.join(eeg_dir, f"{dataset_name}_{i:02d}.edf"))]
        
        if not idx_list:
             print(f"No EDF files found for {model_file} indices.")
             continue

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
        input_channels = sample_batch.shape[1]
        input_time = sample_batch.shape[2]
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        
        # Determine model architecture from state_dict keys
        has_lstm = any('lstm' in k for k in state_dict.keys())
        has_conv = any('conv' in k for k in state_dict.keys())
        
        if has_lstm:
            final_model_type = 'DCNN_BiLSTM'
        elif has_conv:
             final_model_type = 'CNN_MLP'
        else:
             final_model_type = 'MLP'
             
        print(f"  > Inferred architecture: {final_model_type} (from keys)")
        
        model = init_model(final_model_type, input_channels, input_time)
        model.load_state_dict(state_dict)
            
        # Priority: Manual > Training Log > Checkpoint > Default
        threshold = 0.5 # Default
        
        # 1. Manual Override
        # Check both base_name (old behavior) and filename (user preference)
        if model_base_name in MANUAL_THRESHOLDS:
            threshold = MANUAL_THRESHOLDS[model_base_name]
            print(f"Loaded threshold from MANUAL_THRESHOLDS (Base Name): {threshold}")
        elif model_file in MANUAL_THRESHOLDS:
            threshold = MANUAL_THRESHOLDS[model_file]
            print(f"Loaded threshold from MANUAL_THRESHOLDS (File Name): {threshold}")
        else:
            # 2. Try loading from log first
            # We use model_base_name which is cleaned (without _best) and model_type_label (Best Loss vs Best Metric)
            log_threshold = load_threshold_from_log(model_base_name, model_type_label)
            
            if log_threshold is not None:
                threshold = log_threshold
                print(f"Loaded threshold from log ({model_type_label}): {threshold}")
            elif isinstance(checkpoint, dict) and 'threshold' in checkpoint:
                threshold = checkpoint['threshold']
                print(f"Loaded threshold from checkpoint: {threshold}")
            else:
                 print(f"Using default threshold: {threshold}")
        
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
        
        # Initialize Aggregation Variables
        agg_valid_alarms = 0
        agg_missed_count = 0
        agg_fa_pre = 0
        agg_fa_post = 0
        agg_fa_non_sz = 0
        agg_total_false_alarms = 0
        agg_interictal_duration_sec = 0.0
        
        # Denominators for Percentage Calculation
        agg_total_windows_preictal = 0
        agg_total_windows_inter_pre = 0
        agg_total_windows_inter_post = 0
        agg_total_windows_non_sz = 0
        
        agg_last_post_times = []
        
        current_pred_idx = 0
        POINTS_PER_SEC = 256
        SEC_PER_PRED = 5
        
        processed_files = []

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
            # print(f"Saved plot to {save_path}")
            
            processed_files.append(f"{idx:02d}")

            # Calculate Metrics based on new Zone Definitions:
            # Zone 1 (Interictal Pre): Start -> Seizure Start - 15m (False Alarm Zone)
            # Zone 2 (Preictal): Seizure Start - 15m -> Seizure Start (True Alarm Zone)
            # Zone 3 (Ictal + Postictal): Seizure Start -> Seizure End + 15m (Don't Care Zone)
            # Zone 4 (Interictal Post): Seizure End + 15m -> End (False Alarm Zone)
            
            seizure_start = info.get('preictal_end_time')
            seizure_end = info.get('seizure_end_time')
            
            valid_alarms = 0
            missed = False
            fa_pre = 0
            fa_post = 0
            fa_non_sz = 0
            file_interictal_duration = 0.0
            
            if len(file_preds) > 0:
                total_duration_sec = len(file_preds) * SEC_PER_PRED
                
                if seizure_start is not None and seizure_end is not None:
                    # Seizure File
                    preictal_duration = 15 * 60
                    postictal_duration = 15 * 60
                    
                    sz_start_idx = int(seizure_start / SEC_PER_PRED)
                    sz_end_idx = int(seizure_end / SEC_PER_PRED)
                    
                    # Zone 2: Preictal (True Alarm)
                    # valid_start_time = seizure_start - preictal_duration
                    valid_start_idx = int((seizure_start - preictal_duration) / SEC_PER_PRED)
                    valid_alarms = np.sum(file_preds[max(0, valid_start_idx):sz_start_idx])
                    
                    if valid_alarms == 0:
                        missed = True
                    
                    # Zone 1: Interictal Pre (False Alarm)
                    fa_pre = np.sum(file_preds[0:max(0, valid_start_idx)])
                    
                    # Zone 4: Interictal Post (False Alarm)
                    post_interictal_start_time = seizure_end + postictal_duration
                    post_interictal_start_idx = int(post_interictal_start_time / SEC_PER_PRED)
                    
                    if post_interictal_start_idx < len(file_preds):
                        fa_post = np.sum(file_preds[post_interictal_start_idx:])
                        
                    # Accumulate Interictal Duration for FPR
                    # Duration before Preictal
                    z1_duration = max(0, seizure_start - preictal_duration)
                    # Duration after Postictal
                    z4_duration = max(0, total_duration_sec - post_interictal_start_time)
                    file_interictal_duration = z1_duration + z4_duration
                    
                    # Last Post-Seizure Time
                    post_seizure_preds = file_preds[sz_end_idx:]
                    if np.sum(post_seizure_preds) > 0:
                        last_idx = np.where(post_seizure_preds == 1)[0][-1]
                        last_time_sec = (last_idx + 1) * SEC_PER_PRED
                        m, s = divmod(last_time_sec, 60)
                        agg_last_post_times.append(f"{idx:02d}:+{int(m)}m {int(s)}s")

                else:
                    # Interictal File
                    missed = False # Not applicable really, but counts as 0 missed
                    fa_pre = 0
                    fa_post = 0
                    fa_non_sz = np.sum(file_preds)
                    file_interictal_duration = total_duration_sec
            
            # Aggregate
            agg_valid_alarms += valid_alarms
            if missed:
                agg_missed_count += 1
            agg_fa_pre += fa_pre
            agg_fa_post += fa_post
            agg_fa_non_sz += fa_non_sz
            agg_total_false_alarms += (fa_pre + fa_post + fa_non_sz)
            agg_interictal_duration_sec += file_interictal_duration

            # Aggregate Denominators (Total Windows) based on file type
            if seizure_start is not None and seizure_end is not None:
                # Preictal (Zone 2)
                # valid_start_idx to sz_start_idx
                agg_total_windows_preictal += (sz_start_idx - max(0, valid_start_idx))
                
                # Interictal Pre (Zone 1)
                # 0 to valid_start_idx
                agg_total_windows_inter_pre += max(0, valid_start_idx)
                
                # Interictal Post (Zone 4)
                # post_interictal_start_idx to end
                if post_interictal_start_idx < len(file_preds):
                    agg_total_windows_inter_post += (len(file_preds) - post_interictal_start_idx)
            else:
                # Non-Sz File
                agg_total_windows_non_sz += len(file_preds)

        # Final Calculations per Model
        if agg_interictal_duration_sec > 0:
            agg_fpr_h = agg_total_false_alarms / (agg_interictal_duration_sec / 3600)
        else:
            agg_fpr_h = 0.0
            
        last_post_str = ", ".join(agg_last_post_times) if agg_last_post_times else "-"
        files_str = f"{dataset_name}[{','.join(processed_files)}]"

        # Helper format string
        def fmt_stat(count, total):
            if total == 0:
                return "0.0%"
            pct = (count / total) * 100
            return f"{pct:.1f}%"

        str_valid = fmt_stat(agg_valid_alarms, agg_total_windows_preictal)
        str_fa_pre = fmt_stat(agg_fa_pre, agg_total_windows_inter_pre)
        str_fa_post = fmt_stat(agg_fa_post, agg_total_windows_inter_post)
        str_fa_non_sz = fmt_stat(agg_fa_non_sz, agg_total_windows_non_sz)

        # Log to file (Summary Row)
        log_file = os.path.join(base_dir, 'result', 'real_test', 'test_log.md')
        
        # Determine Header based on calculated totals (Use current run's totals as reference)
        header_valid = f"Valid Warnings (/{agg_total_windows_preictal})"
        header_fa_pre = f"FA Pre (/{agg_total_windows_inter_pre})"
        header_fa_post = f"FA Post (/{agg_total_windows_inter_post})"
        header_fa_nonsz = f"FA Non-Sz (/{agg_total_windows_non_sz})"
        
        header = f"| Model Name | Files | {header_valid} | Missed Count | {header_fa_pre} | {header_fa_post} | {header_fa_nonsz} | Total FA | FPR/h | Last Post-Alarm Times |\n|---|---|---|---|---|---|---|---|---|---|\n"
        
        if not os.path.exists(log_file):
            with open(log_file, 'w') as f:
                f.write(header)
        else:
             # If file exists, check if we need to update header (Optional, skipping for simplicity or can implement read-replace)
             # For now, just append. User can delete file to refresh header.
             pass
        
        row = f"| {model_file} | {files_str} | {str_valid} | {agg_missed_count} | {str_fa_pre} | {str_fa_post} | {str_fa_non_sz} | {int(agg_total_false_alarms)} | {agg_fpr_h:.2f} | {last_post_str} |\n"
        with open(log_file, 'a') as f:
            f.write(row)
        print(f"Logged summary to {log_file}")

    print("\nAll models processed.")

