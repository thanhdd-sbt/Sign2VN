"""
Debug Script - Tìm hiểu tại sao chỉ có 973 entries
"""

import os
import pandas as pd
import json
from pathlib import Path


def debug_labeling_results():
    """Debug kết quả labeling để tìm vấn đề"""
    
    print("=" * 80)
    print("DEBUG: LABELING RESULTS")
    print("=" * 80)
    
    # Paths
    meta_csv = '/content/drive/MyDrive/Sign2VN/work/meta.csv'
    landmarks_dir = '/content/drive/MyDrive/Sign2VN/work/landmarks'
    failed_log = '/content/drive/MyDrive/Sign2VN/work/failed_videos.txt'
    stats_file = '/content/drive/MyDrive/Sign2VN/work/extraction_stats.json'
    
    # Check 1: Meta.csv
    print("\n[1] META.CSV")
    print("-" * 80)
    if os.path.exists(meta_csv):
        df = pd.read_csv(meta_csv)
        print(f"✓ Meta.csv exists")
        print(f"  Entries: {len(df)}")
        print(f"  Unique videos: {df['orig_name'].nunique()}")
        print(f"  Unique labels: {df['label_vi'].nunique()}")
        print(f"  Total frames: {df['num_frames'].sum():,}")
        print(f"  Avg frames/video: {df['num_frames'].mean():.1f}")
        
        # Show sample
        print(f"\n  Sample entries:")
        print(df[['orig_name', 'label_vi', 'num_frames']].head(10).to_string(index=False))
    else:
        print("✗ Meta.csv not found!")
        df = None
    
    # Check 2: Landmarks files
    print("\n[2] LANDMARKS FILES")
    print("-" * 80)
    if os.path.exists(landmarks_dir):
        npy_files = [f for f in os.listdir(landmarks_dir) if f.endswith('.npy')]
        print(f"✓ Landmarks directory exists")
        print(f"  .npy files: {len(npy_files)}")
        
        if len(npy_files) > 0:
            print(f"\n  Sample files:")
            for f in npy_files[:10]:
                full_path = os.path.join(landmarks_dir, f)
                size = os.path.getsize(full_path)
                print(f"    {f} ({size:,} bytes)")
    else:
        print("✗ Landmarks directory not found!")
        npy_files = []
    
    # Check 3: Failed videos
    print("\n[3] FAILED VIDEOS")
    print("-" * 80)
    if os.path.exists(failed_log):
        with open(failed_log, 'r') as f:
            failed_content = f.read()
        
        failed_lines = [l for l in failed_content.split('\n') if l.strip() and not l.startswith('=') and not l.startswith('FAILED')]
        print(f"✓ Failed log exists")
        print(f"  Failed entries: ~{len(failed_lines) // 2}")  # Each fail has 2 lines
        
        if len(failed_lines) > 0:
            print(f"\n  Sample failures:")
            for line in failed_lines[:20]:
                if line.strip():
                    print(f"    {line}")
    else:
        print("  No failed videos log")
    
    # Check 4: Stats
    print("\n[4] EXTRACTION STATISTICS")
    print("-" * 80)
    if os.path.exists(stats_file):
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        
        print("✓ Stats file exists")
        
        if 'extraction_stats' in stats:
            ext_stats = stats['extraction_stats']
            print(f"  Total videos: {ext_stats.get('total_videos', 'N/A')}")
            print(f"  Successful: {ext_stats.get('successful', 'N/A')}")
            print(f"  Failed: {ext_stats.get('failed', 'N/A')}")
            print(f"  Total frames: {ext_stats.get('total_frames', 'N/A'):,}")
            
            success_rate = ext_stats.get('successful', 0) / max(ext_stats.get('total_videos', 1), 1) * 100
            print(f"  Success rate: {success_rate:.1f}%")
    else:
        print("  No stats file")
    
    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    
    if df is not None:
        meta_count = len(df)
        npy_count = len(npy_files)
        
        print(f"\nEntries in meta.csv: {meta_count}")
        print(f"Files in landmarks/: {npy_count}")
        
        if meta_count != npy_count:
            print(f"\n⚠️  MISMATCH! meta.csv has {meta_count} but landmarks has {npy_count}")
            
            if meta_count < npy_count:
                print("  → Meta.csv might be incomplete")
            else:
                print("  → Some .npy files might be missing")
        else:
            print("\n✓ Counts match!")
        
        # Check for duplicates
        duplicates = df[df.duplicated(subset=['orig_name'], keep=False)]
        if len(duplicates) > 0:
            print(f"\n⚠️  Found {len(duplicates)} duplicate entries in meta.csv")
            print(duplicates[['orig_name', 'label_vi']].to_string(index=False))
    
    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    if df is not None and len(df) < 1000:
        print("\n⚠️  Only 973 videos processed out of 7200 matched")
        print("\nPossible reasons:")
        print("  1. Most videos failed extraction (check failed_videos.txt)")
        print("  2. Pipeline was interrupted early")
        print("  3. meta.csv was corrupted/truncated")
        print("\nSuggested actions:")
        print("  1. Check failed_videos.txt for error patterns")
        print("  2. Delete meta.csv and re-run pipeline")
        print("  3. Check if landmarks/ has more files than meta.csv")
    else:
        print("\n✓ Results look reasonable")
        print("\nYou can proceed to training!")


def check_landmark_vs_meta():
    """Check consistency between landmarks files and meta.csv"""
    
    print("\n" + "=" * 80)
    print("DETAILED CONSISTENCY CHECK")
    print("=" * 80)
    
    meta_csv = '/content/drive/MyDrive/Sign2VN/work/meta.csv'
    landmarks_dir = '/content/drive/MyDrive/Sign2VN/work/landmarks'
    
    if not os.path.exists(meta_csv) or not os.path.exists(landmarks_dir):
        print("Missing required files")
        return
    
    # Load meta
    df = pd.read_csv(meta_csv)
    meta_files = set(df['orig_name'].tolist())
    
    # Get landmark files
    npy_files = set([f for f in os.listdir(landmarks_dir) if f.endswith('.npy')])
    
    print(f"\nMeta.csv: {len(meta_files)} videos")
    print(f"Landmarks: {len(npy_files)} files")
    
    # Find mismatches
    only_in_meta = []
    for video in meta_files:
        # Try to find corresponding .npy
        found = False
        for npy in npy_files:
            if video in npy or npy in video:
                found = True
                break
        if not found:
            only_in_meta.append(video)
    
    if only_in_meta:
        print(f"\n⚠️  {len(only_in_meta)} videos in meta.csv but no .npy file found:")
        for v in only_in_meta[:10]:
            print(f"    {v}")
    else:
        print("\n✓ All meta.csv entries have corresponding .npy files")


if __name__ == "__main__":
    debug_labeling_results()
    check_landmark_vs_meta()
    
    print("\n" + "=" * 80)
    print("DEBUG COMPLETE")
    print("=" * 80)