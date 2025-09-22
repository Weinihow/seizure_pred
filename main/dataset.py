'''
YCWang
Sept.22, 2025
This code prepare for the train / val / test datasets
'''
import librosa
import numpy as np
from torch.utils.data import Dataset

def eeg_to_mfcc(eeg_segment, sr=256, n_mfcc=13, n_fft=256, hop_length=13):
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
            sr=sr,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length
        )
        mfccs.append(mfcc)
    return np.stack(mfccs)

def sliding_window(eeg_segment, seizure_ontime, duration=10, overlap=2, pre_ict=6, sr=256):
    '''
    This function segment the EEG signals and transform into MFCC
    input : eeg_segment      : shape (n_channels, signal_length) -------- original signal
            seizure_ontime   : seizure onset time (sec)
            duration         : window duration (sec)
            overlap          : window overlap (sec)
            pre_ict          : define pre-ictal range (min)
            sr               : sampling rate
    output: features         : shape (data#, n_channels, n_mfcc, time) -- stack of MFCC data
            labels           : shape (data#, )
    '''
    features = []
    labels = []
    pre_ictal_time = max(0, seizure_ontime - pre_ict*60)

    window_size = duration * sr
    overlap_size = overlap * sr
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
        label = 1 if start >= pre_ictal_time * sr and start <= seizure_ontime * sr else 0
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