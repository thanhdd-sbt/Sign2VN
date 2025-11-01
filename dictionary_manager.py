"""
Dictionary loader and video matcher
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from collections import defaultdict

import labeling_config as config


class DictionaryManager:
    """Quản lý dictionary và match videos với labels"""
    
    def __init__(self, dictionary_path: str):
        self.dictionary_path = dictionary_path
        self.dictionary = []
        self.video_to_label = {}  # video_id -> label info
        self.word_to_videos = defaultdict(list)  # word -> [video_ids]
        
    def load_dictionary(self):
        """Load dictionary từ JSON file"""
        print(f"\nLoading dictionary from: {self.dictionary_path}")
        
        full_path = os.path.join(config.DRIVE_ROOT, self.dictionary_path)
        
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Dictionary not found: {full_path}")
        
        with open(full_path, 'r', encoding='utf-8') as f:
            self.dictionary = json.load(f)
        
        print(f"✓ Loaded {len(self.dictionary)} entries")
        
        # Build lookup tables
        self._build_lookup_tables()
        
        return self
    
    def _build_lookup_tables(self):
        """Xây dựng lookup tables để match nhanh"""
        for entry in self.dictionary:
            video_id = entry.get('_id')
            word = entry.get('word', '')
            
            if video_id:
                self.video_to_label[video_id] = {
                    'word': word,
                    '_word': entry.get('_word', ''),
                    'description': entry.get('description', ''),
                    'type': entry.get('tl', ''),
                    'local_video': entry.get('local_video', ''),
                    'video_url': entry.get('video_url', '')
                }
                
                self.word_to_videos[word].append(video_id)
        
        print(f"✓ Built lookup tables:")
        print(f"  - {len(self.video_to_label)} video IDs")
        print(f"  - {len(self.word_to_videos)} unique words")
    
    def match_video_file(self, video_filename: str) -> Optional[Dict]:
        """
        Match video filename với dictionary entry
        
        Args:
            video_filename: Tên file video (có thể có đường dẫn)
        
        Returns:
            Dictionary entry nếu match, None nếu không
        """
        # Lấy basename
        basename = os.path.basename(video_filename)
        
        # Try exact match with _id
        for entry in self.dictionary:
            if entry.get('local_video', '').endswith(basename):
                return entry
        
        # Try match by _id prefix (e.g., D0001B from D0001B_địa_chỉ.mp4)
        video_id = self._extract_video_id(basename)
        if video_id and video_id in self.video_to_label:
            return self._get_entry_by_id(video_id)
        
        # Try fuzzy match by word in filename
        for word in self.word_to_videos.keys():
            if word in basename:
                video_id = self.word_to_videos[word][0]
                return self._get_entry_by_id(video_id)
        
        return None
    
    def _extract_video_id(self, filename: str) -> Optional[str]:
        """Extract video ID từ filename (e.g., D0001B từ D0001B_địa_chỉ.mp4)"""
        # Remove extension
        name = os.path.splitext(filename)[0]
        
        # Try to extract ID pattern (letter + digits + optional letter)
        import re
        pattern = r'([A-Z]\d{4}[A-Z]?)'
        match = re.search(pattern, name)
        
        if match:
            return match.group(1)
        
        return None
    
    def _get_entry_by_id(self, video_id: str) -> Optional[Dict]:
        """Lấy entry từ dictionary theo video ID"""
        for entry in self.dictionary:
            if entry.get('_id') == video_id:
                return entry
        return None
    
    def get_label_for_video(self, video_filename: str) -> Optional[str]:
        """
        Lấy label (word) cho video
        
        Args:
            video_filename: Tên file video
        
        Returns:
            Label string hoặc None
        """
        entry = self.match_video_file(video_filename)
        if entry:
            return entry.get('word')
        return None
    
    def get_signer_from_filename(self, video_filename: str) -> str:
        """
        Extract signer info từ filename hoặc video_id
        
        Returns:
            Signer identifier (B/N/T) hoặc UNKNOWN
        """
        video_id = self._extract_video_id(video_filename)
        
        if video_id and len(video_id) > 5:
            # Last character might be signer (B, N, T)
            last_char = video_id[-1]
            if last_char in ['B', 'N', 'T']:
                return last_char
        
        return "UNKNOWN"
    
    def get_statistics(self) -> Dict:
        """Lấy thống kê về dictionary"""
        stats = {
            'total_entries': len(self.dictionary),
            'total_video_ids': len(self.video_to_label),
            'total_unique_words': len(self.word_to_videos),
            'words_with_multiple_signers': 0,
            'type_distribution': defaultdict(int)
        }
        
        # Count words with multiple signers
        for word, video_ids in self.word_to_videos.items():
            if len(video_ids) > 1:
                stats['words_with_multiple_signers'] += 1
        
        # Count by type
        for entry in self.dictionary:
            tl = entry.get('tl', 'Unknown')
            stats['type_distribution'][tl] += 1
        
        return stats
    
    def print_statistics(self):
        """In thống kê"""
        stats = self.get_statistics()
        
        print("\n" + "=" * 80)
        print("DICTIONARY STATISTICS")
        print("=" * 80)
        print(f"Total entries: {stats['total_entries']}")
        print(f"Unique video IDs: {stats['total_video_ids']}")
        print(f"Unique words: {stats['total_unique_words']}")
        print(f"Words with multiple signers: {stats['words_with_multiple_signers']}")
        
        print(f"\nType distribution:")
        for tl, count in sorted(stats['type_distribution'].items(), key=lambda x: -x[1]):
            print(f"  {tl}: {count}")
        
        print("=" * 80)
    
    def export_to_dataframe(self) -> pd.DataFrame:
        """Export dictionary to pandas DataFrame"""
        data = []
        for entry in self.dictionary:
            data.append({
                'video_id': entry.get('_id'),
                'word': entry.get('word'),
                '_word': entry.get('_word'),
                'description': entry.get('description'),
                'type': entry.get('tl'),
                'local_video': entry.get('local_video'),
                'video_url': entry.get('video_url')
            })
        
        return pd.DataFrame(data)


def test_dictionary_manager():
    """Test DictionaryManager"""
    print("Testing DictionaryManager...")
    
    dm = DictionaryManager(config.DICTIONARY_PATH)
    dm.load_dictionary()
    dm.print_statistics()
    
    # Test matching
    test_files = [
        "D0001B_địa_chỉ.mp4",
        "D0002_tỉnh.mp4",
        "random_video.mp4"
    ]
    
    print("\nTesting video matching:")
    for filename in test_files:
        label = dm.get_label_for_video(filename)
        signer = dm.get_signer_from_filename(filename)
        print(f"  {filename}")
        print(f"    Label: {label}")
        print(f"    Signer: {signer}")
    
    # Export to DataFrame
    df = dm.export_to_dataframe()
    print(f"\nDataFrame shape: {df.shape}")
    print(df.head())


if __name__ == "__main__":
    test_dictionary_manager()
