"""
Rebuild meta.csv từ landmarks files nếu meta.csv bị corrupt
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import re


def rebuild_meta_from_landmarks():
    """Rebuild meta.csv từ landmarks files"""
    
    print("=" * 80)
    print("REBUILD META.CSV FROM LANDMARKS FILES")
    print("=" * 80)
    
    landmarks_dir = '/content/drive/MyDrive/Sign2VN/work/landmarks'
    meta_csv = '/content/drive/MyDrive/Sign2VN/work/meta.csv'
    
    if not os.path.exists(landmarks_dir):
        print("✗ Landmarks directory not found!")
        return
    
    # Get all .npy files
    npy_files = [f for f in os.listdir(landmarks_dir) if f.endswith('.npy')]
    print(f"\nFound {len(npy_files)} .npy files")
    
    if len(npy_files) == 0:
        print("No .npy files to process!")
        return
    
    # Backup old meta.csv if exists
    if os.path.exists(meta_csv):
        backup_path = meta_csv + '.old'
        os.rename(meta_csv, backup_path)
        print(f"✓ Backed up old meta.csv to: {backup_path}")
    
    # Build new meta
    print("\nProcessing .npy files...")
    entries = []
    
    for i, npy_file in enumerate(npy_files):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(npy_files)}...")
        
        npy_path = os.path.join(landmarks_dir, npy_file)
        
        try:
            # Load landmarks
            landmarks = np.load(npy_path)
            num_frames = len(landmarks)
            
            # Extract info from filename
            # Format: D0001B_địa_chỉ.npy or similar
            basename = os.path.splitext(npy_file)[0]
            
            # Try to extract video_id
            video_id_match = re.search(r'([A-Z]\d{4}[A-Z]?)', basename)
            video_id = video_id_match.group(1) if video_id_match else basename
            
            # Extract label (after video_id and underscore)
            label_vi = basename.replace(video_id, '').lstrip('_')
            if not label_vi:
                label_vi = 'unknown'
            
            # Extract signer
            signer = 'UNKNOWN'
            if len(video_id) > 5:
                last_char = video_id[-1]
                if last_char in ['B', 'N', 'T']:
                    signer = last_char
            
            # Create entry
            entry = {
                'npy': npy_path,
                'label': video_id,
                'label_vi': label_vi,
                'orig_name': f"{video_id}_{label_vi}.mp4",  # Reconstruct filename
                'signer': signer,
                'description': '',
                'type': '',
                'num_frames': num_frames,
                'video_path': ''
            }
            
            entries.append(entry)
            
        except Exception as e:
            print(f"  ✗ Error processing {npy_file}: {e}")
            continue
    
    # Create DataFrame
    df = pd.DataFrame(entries)
    
    # Save
    df.to_csv(meta_csv, index=False, encoding='utf-8')
    
    print(f"\n✓ Created new meta.csv with {len(df)} entries")
    print(f"  Path: {meta_csv}")
    
    # Show stats
    print(f"\nStatistics:")
    print(f"  Total entries: {len(df)}")
    print(f"  Unique labels: {df['label_vi'].nunique()}")
    print(f"  Total frames: {df['num_frames'].sum():,}")
    print(f"  Avg frames/video: {df['num_frames'].mean():.1f}")
    
    print(f"\nSigner distribution:")
    print(df['signer'].value_counts())
    
    print(f"\nSample entries:")
    print(df[['orig_name', 'label_vi', 'num_frames', 'signer']].head(10).to_string(index=False))
    
    return df


def verify_rebuilt_meta():
    """Verify rebuilt meta.csv"""
    
    print("\n" + "=" * 80)
    print("VERIFICATION")
    print("=" * 80)
    
    meta_csv = '/content/drive/MyDrive/Sign2VN/work/meta.csv'
    
    if not os.path.exists(meta_csv):
        print("✗ meta.csv not found!")
        return False
    
    df = pd.read_csv(meta_csv)
    
    # Check all .npy files exist
    missing = []
    for i, row in df.iterrows():
        if not os.path.exists(row['npy']):
            missing.append(row['npy'])
    
    if missing:
        print(f"⚠️  {len(missing)} .npy files referenced but not found:")
        for m in missing[:5]:
            print(f"    {m}")
        return False
    else:
        print(f"✓ All {len(df)} .npy files exist and accessible")
    
    # Check for duplicates
    duplicates = df[df.duplicated(subset=['orig_name'], keep=False)]
    if len(duplicates) > 0:
        print(f"⚠️  {len(duplicates)} duplicate entries found")
        return False
    else:
        print("✓ No duplicate entries")
    
    # Check frame counts
    zero_frames = df[df['num_frames'] == 0]
    if len(zero_frames) > 0:
        print(f"⚠️  {len(zero_frames)} videos with 0 frames")
        return False
    else:
        print("✓ All videos have > 0 frames")
    
    print("\n✅ Meta.csv looks good!")
    return True


if __name__ == "__main__":
    df = rebuild_meta_from_landmarks()
    
    if df is not None:
        verify_rebuilt_meta()
        
        print("\n" + "=" * 80)
        print("NEXT STEPS")
        print("=" * 80)
        print("\n1. Verify meta.csv looks correct")
        print("2. If good, start training:")
        print("   python train.py --num_epochs 100 --batch_size 16 --test")
        print("\n3. If you want to process more videos:")
        print("   pipeline.run(resume=True)")