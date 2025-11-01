"""
Data loading and preprocessing utilities for Sign2VN
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional
import random
from collections import Counter

import config


class VietnameseTokenizer:
    """Tokenizer cho tiếng Việt"""
    
    def __init__(self):
        self.word2idx = {
            config.PAD_TOKEN: 0,
            config.SOS_TOKEN: 1,
            config.EOS_TOKEN: 2,
            config.UNK_TOKEN: 3
        }
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)
    
    def build_vocab(self, sentences: List[str], min_freq: int = 2):
        """Xây dựng vocabulary từ danh sách câu"""
        word_freq = Counter()
        for sentence in sentences:
            words = sentence.lower().split()
            word_freq.update(words)
        
        # Thêm các từ có tần suất >= min_freq
        for word, freq in word_freq.items():
            if freq >= min_freq and word not in self.word2idx:
                self.word2idx[word] = len(self.word2idx)
                self.idx2word[len(self.idx2word)] = word
        
        self.vocab_size = len(self.word2idx)
        print(f"Vocabulary size: {self.vocab_size}")
        return self
    
    def encode(self, sentence: str, max_length: Optional[int] = None) -> List[int]:
        """Chuyển câu thành sequence of indices"""
        words = sentence.lower().split()
        indices = [self.word2idx[config.SOS_TOKEN]]
        
        for word in words:
            idx = self.word2idx.get(word, self.word2idx[config.UNK_TOKEN])
            indices.append(idx)
        
        indices.append(self.word2idx[config.EOS_TOKEN])
        
        # Padding hoặc truncate
        if max_length:
            if len(indices) < max_length:
                indices.extend([self.word2idx[config.PAD_TOKEN]] * (max_length - len(indices)))
            else:
                indices = indices[:max_length-1] + [self.word2idx[config.EOS_TOKEN]]
        
        return indices
    
    def decode(self, indices: List[int], skip_special: bool = True) -> str:
        """Chuyển indices thành câu"""
        words = []
        for idx in indices:
            word = self.idx2word.get(idx, config.UNK_TOKEN)
            if skip_special and word in [config.PAD_TOKEN, config.SOS_TOKEN, config.EOS_TOKEN]:
                if word == config.EOS_TOKEN:
                    break
                continue
            words.append(word)
        return ' '.join(words)


class SignLanguageDataset(Dataset):
    """Dataset cho Sign Language Recognition"""
    
    def __init__(
        self, 
        npy_paths: List[str], 
        labels: List[str],
        tokenizer: VietnameseTokenizer,
        max_seq_length: int = config.MAX_SEQUENCE_LENGTH,
        augment: bool = False
    ):
        self.npy_paths = npy_paths
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.augment = augment
    
    def __len__(self):
        return len(self.npy_paths)
    
    def augment_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """Data augmentation cho sequence"""
        if not self.augment or random.random() > config.AUGMENTATION_PROB:
            return sequence
        
        aug_sequence = sequence.copy()
        
        # 1. Thêm nhiễu Gaussian
        if random.random() > 0.5:
            noise = np.random.normal(0, config.NOISE_SCALE, aug_sequence.shape)
            aug_sequence += noise
        
        # 2. Time warping (thay đổi tốc độ)
        if random.random() > 0.5:
            T = len(aug_sequence)
            new_T = int(T * (1 + random.uniform(-config.TIME_WARPING_PARAM, config.TIME_WARPING_PARAM)))
            new_T = max(config.MIN_SEQUENCE_LENGTH, min(new_T, self.max_seq_length))
            
            # Interpolate
            indices = np.linspace(0, T-1, new_T)
            aug_sequence = np.array([
                np.interp(indices, np.arange(T), aug_sequence[:, i])
                for i in range(aug_sequence.shape[1])
            ]).T
        
        # 3. Horizontal flip (đảo trái phải cho hands)
        if random.random() > 0.7:
            # Swap left and right hand landmarks
            pose_end = config.NUM_POSE_LANDMARKS * config.LANDMARK_DIM
            left_hand_start = pose_end
            left_hand_end = left_hand_start + config.NUM_HAND_LANDMARKS * config.LANDMARK_DIM
            right_hand_start = left_hand_end
            right_hand_end = right_hand_start + config.NUM_HAND_LANDMARKS * config.LANDMARK_DIM
            
            left_hand = aug_sequence[:, left_hand_start:left_hand_end].copy()
            right_hand = aug_sequence[:, right_hand_start:right_hand_end].copy()
            
            aug_sequence[:, left_hand_start:left_hand_end] = right_hand
            aug_sequence[:, right_hand_start:right_hand_end] = left_hand
            
            # Flip x coordinates
            for i in range(0, aug_sequence.shape[1], config.LANDMARK_DIM):
                aug_sequence[:, i] = 1 - aug_sequence[:, i]
        
        return aug_sequence
    
    def normalize_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """Normalize sequence về cùng scale"""
        # Center around mean và scale về [-1, 1]
        mean = np.mean(sequence, axis=0)
        std = np.std(sequence, axis=0)
        std = np.where(std == 0, 1, std)  # Tránh chia cho 0
        
        normalized = (sequence - mean) / std
        return normalized
    
    def pad_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """Pad hoặc truncate sequence về max_seq_length"""
        T = len(sequence)
        
        if T > self.max_seq_length:
            # Truncate
            return sequence[:self.max_seq_length]
        elif T < self.max_seq_length:
            # Pad với zeros
            pad_length = self.max_seq_length - T
            padding = np.zeros((pad_length, sequence.shape[1]))
            return np.vstack([sequence, padding])
        else:
            return sequence
    
    def __getitem__(self, idx):
        # Load landmarks
        try:
            landmarks = np.load(self.npy_paths[idx])
        except Exception as e:
            print(f"Error loading {self.npy_paths[idx]}: {e}")
            # Return dummy data
            landmarks = np.zeros((config.MIN_SEQUENCE_LENGTH, config.TOTAL_LANDMARKS * config.LANDMARK_DIM))
        
        # Preprocess
        landmarks = self.augment_sequence(landmarks)
        landmarks = self.normalize_sequence(landmarks)
        landmarks = self.pad_sequence(landmarks)
        
        # Get actual length (trước khi pad)
        actual_length = min(len(landmarks), self.max_seq_length)
        
        # Tokenize label
        label_vi = self.labels[idx]
        label_tokens = self.tokenizer.encode(label_vi)
        
        return {
            'landmarks': torch.FloatTensor(landmarks),
            'landmarks_length': torch.LongTensor([actual_length]),
            'label_tokens': torch.LongTensor(label_tokens),
            'label_text': label_vi
        }


def load_and_split_data(
    meta_csv_path: str,
    tokenizer: VietnameseTokenizer,
    val_split: float = config.VALIDATION_SPLIT,
    test_split: float = config.TEST_SPLIT,
    random_seed: int = 42
) -> Tuple[SignLanguageDataset, SignLanguageDataset, SignLanguageDataset]:
    """
    Load data từ meta.csv và split thành train/val/test sets
    
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    # Load meta.csv
    df = pd.read_csv(meta_csv_path)
    print(f"Loaded {len(df)} samples from {meta_csv_path}")
    
    # Lấy paths và labels
    npy_paths = df['npy'].tolist()
    labels_vi = df['label_vi'].tolist()
    
    # Build vocabulary
    tokenizer.build_vocab(labels_vi, min_freq=1)
    
    # Split data
    # Đầu tiên split ra test set
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        npy_paths, labels_vi,
        test_size=test_split,
        random_state=random_seed,
        shuffle=True
    )
    
    # Sau đó split train thành train và val
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels,
        test_size=val_split / (1 - test_split),
        random_state=random_seed,
        shuffle=True
    )
    
    print(f"Train samples: {len(train_paths)}")
    print(f"Val samples: {len(val_paths)}")
    print(f"Test samples: {len(test_paths)}")
    
    # Create datasets
    train_dataset = SignLanguageDataset(
        train_paths, train_labels, tokenizer, 
        max_seq_length=config.MAX_SEQUENCE_LENGTH,
        augment=True
    )
    
    val_dataset = SignLanguageDataset(
        val_paths, val_labels, tokenizer,
        max_seq_length=config.MAX_SEQUENCE_LENGTH,
        augment=False
    )
    
    test_dataset = SignLanguageDataset(
        test_paths, test_labels, tokenizer,
        max_seq_length=config.MAX_SEQUENCE_LENGTH,
        augment=False
    )
    
    return train_dataset, val_dataset, test_dataset


def collate_fn(batch):
    """Custom collate function cho DataLoader"""
    landmarks = torch.stack([item['landmarks'] for item in batch])
    landmarks_lengths = torch.cat([item['landmarks_length'] for item in batch])
    
    # Pad label tokens
    max_label_len = max([len(item['label_tokens']) for item in batch])
    label_tokens = []
    for item in batch:
        tokens = item['label_tokens']
        if len(tokens) < max_label_len:
            # Pad
            padded = torch.cat([
                tokens,
                torch.LongTensor([0] * (max_label_len - len(tokens)))
            ])
        else:
            padded = tokens[:max_label_len]
        label_tokens.append(padded)
    
    label_tokens = torch.stack(label_tokens)
    label_texts = [item['label_text'] for item in batch]
    
    return {
        'landmarks': landmarks,
        'landmarks_lengths': landmarks_lengths,
        'label_tokens': label_tokens,
        'label_texts': label_texts
    }


def create_dataloaders(
    train_dataset: SignLanguageDataset,
    val_dataset: SignLanguageDataset,
    test_dataset: SignLanguageDataset,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = 2
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Tạo DataLoaders cho train/val/test"""
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
