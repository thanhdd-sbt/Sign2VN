"""
Video scanner - Scan videos từ shared folders
"""

import os
from pathlib import Path
from typing import List, Dict
import pandas as pd

import labeling_config as config


class VideoScanner:
    """Scan videos từ shared folders"""
    
    def __init__(self, shared_folders: List[str]):
        self.shared_folders = shared_folders
        self.videos = []
        
    def scan_videos(self) -> List[Dict]:
        """
        Scan tất cả videos từ shared folders
        
        Returns:
            List of video info dicts
        """
        print("\n" + "=" * 80)
        print("SCANNING VIDEOS")
        print("=" * 80)
        
        all_videos = []
        
        for folder in self.shared_folders:
            print(f"\nScanning folder: {folder}")
            full_path = os.path.join(config.DRIVE_ROOT, folder)
            
            if not os.path.exists(full_path):
                print(f"  ⚠️  Folder not found: {full_path}")
                print(f"     Skipping...")
                continue
            
            videos = self._scan_folder(full_path)
            print(f"  ✓ Found {len(videos)} videos")
            
            all_videos.extend(videos)
        
        self.videos = all_videos
        
        print("\n" + "=" * 80)
        print(f"TOTAL VIDEOS FOUND: {len(all_videos)}")
        print("=" * 80)
        
        return all_videos
    
    def _scan_folder(self, folder_path: str) -> List[Dict]:
        """Scan một folder và subfolders"""
        videos = []
        
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                # Check extension
                ext = os.path.splitext(file)[1].lower()
                if ext in config.VIDEO_EXTENSIONS:
                    video_path = os.path.join(root, file)
                    
                    video_info = {
                        'filename': file,
                        'full_path': video_path,
                        'relative_path': os.path.relpath(video_path, config.DRIVE_ROOT),
                        'folder': os.path.basename(root),
                        'extension': ext
                    }
                    
                    videos.append(video_info)
        
        return videos
    
    def get_statistics(self) -> Dict:
        """Lấy thống kê về videos"""
        if not self.videos:
            return {}
        
        stats = {
            'total_videos': len(self.videos),
            'by_folder': {},
            'by_extension': {}
        }
        
        # Count by folder
        for video in self.videos:
            folder = video['folder']
            stats['by_folder'][folder] = stats['by_folder'].get(folder, 0) + 1
        
        # Count by extension
        for video in self.videos:
            ext = video['extension']
            stats['by_extension'][ext] = stats['by_extension'].get(ext, 0) + 1
        
        return stats
    
    def print_statistics(self):
        """In thống kê"""
        stats = self.get_statistics()
        
        print("\n" + "=" * 80)
        print("VIDEO STATISTICS")
        print("=" * 80)
        print(f"Total videos: {stats['total_videos']}")
        
        print(f"\nBy folder:")
        for folder, count in sorted(stats['by_folder'].items()):
            print(f"  {folder}: {count}")
        
        print(f"\nBy extension:")
        for ext, count in sorted(stats['by_extension'].items()):
            print(f"  {ext}: {count}")
        
        print("=" * 80)
    
    def export_to_dataframe(self) -> pd.DataFrame:
        """Export videos to DataFrame"""
        return pd.DataFrame(self.videos)
    
    def filter_videos(self, pattern: str = None) -> List[Dict]:
        """
        Filter videos by pattern
        
        Args:
            pattern: Pattern to match in filename
        
        Returns:
            Filtered list of videos
        """
        if not pattern:
            return self.videos
        
        return [v for v in self.videos if pattern in v['filename']]


def test_video_scanner():
    """Test VideoScanner"""
    print("Testing VideoScanner...")
    
    scanner = VideoScanner(config.SHARED_FOLDERS)
    videos = scanner.scan_videos()
    scanner.print_statistics()
    
    # Show sample videos
    print("\nSample videos (first 10):")
    for i, video in enumerate(videos[:10]):
        print(f"{i+1}. {video['filename']}")
        print(f"   Path: {video['relative_path']}")
    
    # Export to DataFrame
    df = scanner.export_to_dataframe()
    print(f"\nDataFrame shape: {df.shape}")
    print(df.head())


if __name__ == "__main__":
    test_video_scanner()
