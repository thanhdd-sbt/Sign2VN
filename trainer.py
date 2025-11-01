"""
Training script for Sign2Text model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import os
import numpy as np
from collections import defaultdict
import json
from datetime import datetime

import config
from model import Sign2TextModel
from data_loader import VietnameseTokenizer


class Trainer:
    """Trainer class cho Sign2Text model"""
    
    def __init__(
        self,
        model: Sign2TextModel,
        tokenizer: VietnameseTokenizer,
        train_loader,
        val_loader,
        learning_rate: float = config.LEARNING_RATE,
        checkpoint_dir: str = config.CHECKPOINT_DIR
    ):
        self.model = model.to(config.DEVICE)
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpoint_dir = checkpoint_dir
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Loss function (ignore padding index)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=config.REDUCE_LR_PATIENCE,
            verbose=True
        )
        
        # Training history
        self.history = defaultdict(list)
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        self.start_epoch = 0
    
    def compute_accuracy(self, predictions, targets):
        """
        Tính accuracy (bỏ qua padding tokens)
        Args:
            predictions: (batch, seq_len, vocab_size)
            targets: (batch, seq_len)
        """
        # Get predicted tokens
        pred_tokens = predictions.argmax(dim=-1)
        
        # Create mask for non-padding tokens
        mask = (targets != 0)
        
        # Compute accuracy
        correct = (pred_tokens == targets) & mask
        accuracy = correct.sum().item() / mask.sum().item()
        
        return accuracy
    
    def compute_bleu(self, predictions, targets):
        """
        Tính BLEU score (simplified version)
        """
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        
        pred_tokens = predictions.argmax(dim=-1)
        
        bleu_scores = []
        for pred, target in zip(pred_tokens, targets):
            # Decode to text
            pred_text = self.tokenizer.decode(pred.cpu().tolist())
            target_text = self.tokenizer.decode(target.cpu().tolist())
            
            # Tokenize
            pred_words = pred_text.split()
            target_words = target_text.split()
            
            # Compute BLEU
            if len(pred_words) > 0 and len(target_words) > 0:
                smooth = SmoothingFunction()
                score = sentence_bleu(
                    [target_words], 
                    pred_words,
                    smoothing_function=smooth.method1
                )
                bleu_scores.append(score)
        
        return np.mean(bleu_scores) if bleu_scores else 0.0
    
    def train_epoch(self, epoch, teacher_forcing_ratio=0.5):
        """Train for one epoch"""
        self.model.train()
        
        epoch_loss = 0
        epoch_acc = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            landmarks = batch['landmarks'].to(config.DEVICE)
            lengths = batch['landmarks_lengths'].squeeze().to(config.DEVICE)
            target_tokens = batch['label_tokens'].to(config.DEVICE)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            outputs = self.model(
                landmarks,
                lengths,
                target_tokens,
                teacher_forcing_ratio=teacher_forcing_ratio
            )
            
            # Compute loss (bỏ qua <SOS> token ở targets)
            # outputs: (batch, seq_len-1, vocab_size)
            # targets: (batch, seq_len)
            targets = target_tokens[:, 1:]  # Bỏ <SOS>
            
            # Reshape for loss computation
            loss = self.criterion(
                outputs.reshape(-1, self.model.vocab_size),
                targets.reshape(-1)
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Compute accuracy
            acc = self.compute_accuracy(outputs, targets)
            
            # Update metrics
            epoch_loss += loss.item()
            epoch_acc += acc
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{acc:.4f}'
            })
        
        avg_loss = epoch_loss / num_batches
        avg_acc = epoch_acc / num_batches
        
        return avg_loss, avg_acc
    
    def validate(self, epoch):
        """Validate model"""
        self.model.eval()
        
        epoch_loss = 0
        epoch_acc = 0
        epoch_bleu = 0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")
            
            for batch in pbar:
                landmarks = batch['landmarks'].to(config.DEVICE)
                lengths = batch['landmarks_lengths'].squeeze().to(config.DEVICE)
                target_tokens = batch['label_tokens'].to(config.DEVICE)
                
                # Forward pass (no teacher forcing)
                outputs = self.model(
                    landmarks,
                    lengths,
                    target_tokens,
                    teacher_forcing_ratio=0.0
                )
                
                targets = target_tokens[:, 1:]
                
                # Compute loss
                loss = self.criterion(
                    outputs.reshape(-1, self.model.vocab_size),
                    targets.reshape(-1)
                )
                
                # Compute metrics
                acc = self.compute_accuracy(outputs, targets)
                
                try:
                    bleu = self.compute_bleu(outputs, targets)
                except:
                    bleu = 0.0
                
                epoch_loss += loss.item()
                epoch_acc += acc
                epoch_bleu += bleu
                num_batches += 1
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{acc:.4f}',
                    'bleu': f'{bleu:.4f}'
                })
        
        avg_loss = epoch_loss / num_batches
        avg_acc = epoch_acc / num_batches
        avg_bleu = epoch_bleu / num_batches
        
        return avg_loss, avg_acc, avg_bleu
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': dict(self.history)
        }
        
        # Save latest checkpoint
        latest_path = os.path.join(self.checkpoint_dir, 'latest_checkpoint.pt')
        torch.save(checkpoint, latest_path)
        print(f"Saved checkpoint to {latest_path}")
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")
        
        # Save periodic checkpoint
        if epoch % config.SAVE_CHECKPOINT_EVERY == 0:
            epoch_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
            torch.save(checkpoint, epoch_path)
    
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint"""
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = defaultdict(list, checkpoint['history'])
        self.start_epoch = checkpoint['epoch'] + 1
        
        print(f"Resumed from epoch {self.start_epoch}")
    
    def train(self, num_epochs=config.NUM_EPOCHS, resume_from=None):
        """
        Main training loop
        Args:
            num_epochs: number of epochs to train
            resume_from: path to checkpoint to resume from
        """
        if resume_from and os.path.exists(resume_from):
            self.load_checkpoint(resume_from)
        
        print(f"\nStarting training from epoch {self.start_epoch}")
        print(f"Device: {config.DEVICE}")
        print(f"Total epochs: {num_epochs}")
        print(f"Batch size: {config.BATCH_SIZE}")
        print(f"Learning rate: {config.LEARNING_RATE}")
        print("=" * 80)
        
        for epoch in range(self.start_epoch, num_epochs):
            # Adjust teacher forcing ratio (decay over time)
            teacher_forcing_ratio = max(0.5 * (0.95 ** epoch), 0.1)
            
            # Train
            train_loss, train_acc = self.train_epoch(epoch, teacher_forcing_ratio)
            
            # Validate
            val_loss, val_acc, val_bleu = self.validate(epoch)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_bleu'].append(val_bleu)
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            # Print epoch summary
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f} | Val BLEU: {val_bleu:.4f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            print(f"  Teacher Forcing Ratio: {teacher_forcing_ratio:.4f}")
            
            # Check for improvement
            is_best = val_loss < self.best_val_loss
            if is_best:
                print(f"  ✓ New best model! (prev: {self.best_val_loss:.4f})")
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1
                print(f"  No improvement for {self.epochs_no_improve} epochs")
            
            # Save checkpoint
            self.save_checkpoint(epoch, is_best)
            
            # Early stopping
            if self.epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
            
            print("=" * 80)
        
        # Save final history
        history_path = os.path.join(self.checkpoint_dir, 'training_history.json')
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)
        
        print(f"\nTraining completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Training history saved to {history_path}")
    
    def test(self, test_loader):
        """Test model on test set"""
        print("\n" + "=" * 80)
        print("Testing model on test set")
        print("=" * 80)
        
        self.model.eval()
        
        test_loss = 0
        test_acc = 0
        test_bleu = 0
        num_batches = 0
        
        # Sample predictions for visualization
        sample_predictions = []
        
        with torch.no_grad():
            pbar = tqdm(test_loader, desc="Testing")
            
            for batch_idx, batch in enumerate(pbar):
                landmarks = batch['landmarks'].to(config.DEVICE)
                lengths = batch['landmarks_lengths'].squeeze().to(config.DEVICE)
                target_tokens = batch['label_tokens'].to(config.DEVICE)
                target_texts = batch['label_texts']
                
                # Generate predictions
                generated_tokens, _ = self.model.generate(
                    landmarks,
                    lengths,
                    max_length=50,
                    sos_token=self.tokenizer.word2idx[config.SOS_TOKEN],
                    eos_token=self.tokenizer.word2idx[config.EOS_TOKEN]
                )
                
                # Compute metrics using teacher forcing for fair comparison
                outputs = self.model(landmarks, lengths, target_tokens, teacher_forcing_ratio=0.0)
                targets = target_tokens[:, 1:]
                
                loss = self.criterion(
                    outputs.reshape(-1, self.model.vocab_size),
                    targets.reshape(-1)
                )
                
                acc = self.compute_accuracy(outputs, targets)
                
                try:
                    bleu = self.compute_bleu(outputs, targets)
                except:
                    bleu = 0.0
                
                test_loss += loss.item()
                test_acc += acc
                test_bleu += bleu
                num_batches += 1
                
                # Save sample predictions
                if batch_idx < 5:  # Save first 5 batches
                    for i in range(min(3, len(generated_tokens))):  # 3 samples per batch
                        pred_text = self.tokenizer.decode(generated_tokens[i].cpu().tolist())
                        target_text = target_texts[i]
                        sample_predictions.append({
                            'target': target_text,
                            'prediction': pred_text
                        })
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{acc:.4f}',
                    'bleu': f'{bleu:.4f}'
                })
        
        avg_test_loss = test_loss / num_batches
        avg_test_acc = test_acc / num_batches
        avg_test_bleu = test_bleu / num_batches
        
        print(f"\nTest Results:")
        print(f"  Loss: {avg_test_loss:.4f}")
        print(f"  Accuracy: {avg_test_acc:.4f}")
        print(f"  BLEU Score: {avg_test_bleu:.4f}")
        
        print(f"\nSample Predictions:")
        for i, sample in enumerate(sample_predictions[:10]):
            print(f"\n  Sample {i+1}:")
            print(f"    Target:     {sample['target']}")
            print(f"    Prediction: {sample['prediction']}")
        
        # Save test results
        test_results = {
            'test_loss': avg_test_loss,
            'test_acc': avg_test_acc,
            'test_bleu': avg_test_bleu,
            'sample_predictions': sample_predictions,
            'timestamp': datetime.now().isoformat()
        }
        
        results_path = os.path.join(self.checkpoint_dir, 'test_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(test_results, f, indent=2, ensure_ascii=False)
        
        print(f"\nTest results saved to {results_path}")
        
        return avg_test_loss, avg_test_acc, avg_test_bleu
