'''
YCWang
Sept.22, 2025
This code prepare for the train / val / test datasets
'''
import librosa
import numpy as np
from torch.utils.data import Dataset

def eeg_to_mfcc(eeg_segment, fs=256, n_mfcc=13, n_fft=256, hop_length=13):
    '''
    This function transform EEG segmants to MFCC
    input : eeg_segment: shape (n_channels, signal_length)
    output: shape (n_channels, n_mfcc, time)
    '''
    mfccs = []
    for channel_data in eeg_segment:
        channel_data = np.asarray(channel_data, dtype=np.float32)
        mfcc = librosa.feature.mfcc(
            y=np.asarray(channel_data, dtype=np.float32),
            fs=fs,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length
        )
        mfccs.append(mfcc)
    return np.stack(mfccs)

def sliding_window(eeg_segment, seizure_ontime, duration=10, overlap=2, pre_ict=6, fs=256):
    '''
    This function segment the EEG signals and transform into MFCC
    input : eeg_segment      : shape (n_channels, signal_length) -------- original signal
            seizure_ontime   : seizure onset time (sec)
            duration         : window duration (sec)
            overlap          : window overlap (sec)
            pre_ict          : define pre-ictal range (min)
            fs               : sampling rate
    output: features         : shape (data#, n_channels, n_mfcc, time) -- stack of MFCC data
            labels           : shape (data#, )
    '''
    features = []
    labels = []
    pre_ictal_time = max(0, seizure_ontime - pre_ict*60)

    window_size = duration * fs
    overlap_size = overlap * fs
    stride = window_size - overlap_size
    total_time = eeg_segment.shape[1]

    # data_start = max(0, (seizure_ontime - pre_ict*60*2)*sampling_rate)
    # data_end = min(total_time, (seizure_ontime + pre_ict*60*3)*sampling_rate)
    # print(data_start, data_end)
    # for start in range(data_start, data_end, stride):
    for start in range(0, total_time, stride):
        end = start + window_size
        # print(start, end)
        segment = eeg_segment[:, start:end]
        if segment.shape[1] != window_size:
          continue
        mfcc = eeg_to_mfcc(segment)
        label = 1 if start >= pre_ictal_time * fs and start <= seizure_ontime * fs else 0
        features.append(mfcc)
        labels.append(label)

    return np.stack(features), np.array(labels)

class MFCCDataset(Dataset):
    def __init__(self, eeg_signal, seizure_ontime):
        '''
        eeg_signal: stack of raw signals
        seizure_ontime: array of seizure on-time
        '''
        try:
            self.data = []
            self.label = []
            for i, segment in enumerate(eeg_signal):
                data, label = sliding_window(
                                eeg_segment=segment,
                                seizure_ontime=seizure_ontime[i]
                            )
                self.data.extend(data)
                self.label.extend(label)
            if len(self.data) != len(self.label):
                print('data number and label number mismatch')
                raise 
        except:
            print('dataset loading error')
            raise
        
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]
    
def raw_window(segment, info, duration=5, fs=256):
    '''
    segment: raw eeg segments (shape: (n_channel, signal_length))
    info: dict with "interictal_start / end time" / "preictal_start / end time"
    '''
    data = []
    label = []
    samples_per_segment = fs * duration

    if "interictal_start_time" in info:
        start = info['interictal_start_time']*fs
        end = info['interictal_end_time']*fs
        trimmed_signal = segment[start:end]

        n_segments = len(trimmed_signal) // samples_per_segment

        n_data = []
        n_label = []
        for i in range(n_segments):
            start = i * samples_per_segment
            end = start + samples_per_segment
            n_data.append(trimmed_signal[start:end])
            n_label.append(0)
        
        data.extend(n_data)
        label.extend(n_label)

    if "preictal_start_time" in info:
        start = info['preictal_start_time']*fs
        end = info['seizure_start_time']*fs
        trimmed_signal = segment[start:end]

        n_segments = len(trimmed_signal) // samples_per_segment

        n_data = []
        n_label = []
        for i in range(n_segments):
            start = i * samples_per_segment
            end = start + samples_per_segment
            n_data.append(trimmed_signal[start:end])
            n_label.append(1)
        
        data.extend(n_data)
        label.extend(n_label)

    return data, label
    
class RawDataset(Dataset):
    def __init__(self, eeg_signal, timepoints):
        try:
            self.data = []
            self.label = []
            for i, segment in enumerate(eeg_signal):
                _, info = timepoints[i]
                data, label = raw_window(segment=segment, info=info, duration=5, fs=256)
                self.data.extend(data)
                self.label.extend(label)
            if len(self.data) != len(self.label):
                print('data number and label number mismatch')
                raise 
        except:
            print('dataset loading error')
            raise
    
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]