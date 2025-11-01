"""
Main training script
Chạy file này để train model từ đầu hoặc resume training
"""

import os
import sys
import torch
import argparse

# Mount Google Drive
try:
    from google.colab import drive
    drive.mount('/content/drive')
    print("Google Drive mounted successfully!")
except:
    print("Not running in Colab or Drive already mounted")

import config
from data_loader import (
    VietnameseTokenizer,
    load_and_split_data,
    create_dataloaders
)
from model import Sign2TextModel
from trainer import Trainer


def main(args):
    print("=" * 80)
    print("SIGN LANGUAGE TO VIETNAMESE TEXT - TRAINING")
    print("=" * 80)
    
    # 1. Khởi tạo tokenizer
    print("\n1. Initializing tokenizer...")
    tokenizer = VietnameseTokenizer()
    
    # 2. Load và split data
    print("\n2. Loading and splitting data...")
    train_dataset, val_dataset, test_dataset = load_and_split_data(
        meta_csv_path=config.META_CSV_PATH,
        tokenizer=tokenizer,
        val_split=config.VALIDATION_SPLIT,
        test_split=config.TEST_SPLIT,
        random_seed=args.seed
    )
    
    # 3. Tạo dataloaders
    print("\n3. Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    
    # 4. Khởi tạo model
    print("\n4. Initializing model...")
    model = Sign2TextModel(vocab_size=tokenizer.vocab_size)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # 5. Khởi tạo trainer
    print("\n5. Initializing trainer...")
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.learning_rate,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # 6. Train model
    print("\n6. Starting training...")
    trainer.train(
        num_epochs=args.num_epochs,
        resume_from=args.resume_from
    )
    
    # 7. Test model
    if args.test:
        print("\n7. Testing model...")
        trainer.test(test_loader)
    
    # 8. Lưu tokenizer
    print("\n8. Saving tokenizer...")
    import pickle
    tokenizer_path = os.path.join(args.checkpoint_dir, 'tokenizer.pkl')
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)
    print(f"   Tokenizer saved to {tokenizer_path}")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETED!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Sign2Text Model")
    
    # Data arguments
    parser.add_argument('--meta_csv', type=str, default=config.META_CSV_PATH,
                        help='Path to meta.csv file')
    
    # Training arguments
    parser.add_argument('--num_epochs', type=int, default=config.NUM_EPOCHS,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=config.LEARNING_RATE,
                        help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of data loading workers')
    
    # Checkpoint arguments
    parser.add_argument('--checkpoint_dir', type=str, default=config.CHECKPOINT_DIR,
                        help='Directory to save checkpoints')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--test', action='store_true',
                        help='Run test after training')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    main(args)
