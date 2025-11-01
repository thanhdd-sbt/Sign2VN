"""
Main Data Labeling Pipeline
Scan videos, extract landmarks, create labels
"""

import os
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import hashlib

import labeling_config as config
from dictionary_manager import DictionaryManager
from video_scanner import VideoScanner
from landmark_extractor import LandmarkExtractor


class DataLabelingPipeline:
    """Pipeline hoàn chỉnh cho data labeling"""
    
    def __init__(self):
        self.dictionary_manager = None
        self.video_scanner = None
        self.landmark_extractor = LandmarkExtractor()
        self.processed_videos = []
        self.meta_data = []
        
    def run(self, resume: bool = True):
        """
        Chạy pipeline đầy đủ
        
        Args:
            resume: Tiếp tục từ checkpoint nếu có
        """
        print("\n" + "=" * 80)
        print("SIGN LANGUAGE DATA LABELING PIPELINE")
        print("=" * 80)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Step 1: Load dictionary
        print("\n[Step 1/5] Loading dictionary...")
        self.dictionary_manager = DictionaryManager(config.DICTIONARY_PATH)
        self.dictionary_manager.load_dictionary()
        self.dictionary_manager.print_statistics()
        
        # Step 2: Scan videos
        print("\n[Step 2/5] Scanning videos...")
        self.video_scanner = VideoScanner(config.SHARED_FOLDERS)
        videos = self.video_scanner.scan_videos()
        self.video_scanner.print_statistics()
        
        if len(videos) == 0:
            print("\n✗ No videos found! Please check paths.")
            return
        
        # Step 3: Match videos with dictionary
        print("\n[Step 3/5] Matching videos with dictionary...")
        matched_videos = self._match_videos_with_dictionary(videos)
        
        print(f"\nMatching results:")
        print(f"  Total videos: {len(videos)}")
        print(f"  Matched: {len(matched_videos)}")
        print(f"  Unmatched: {len(videos) - len(matched_videos)}")
        
        if len(matched_videos) == 0:
            print("\n✗ No videos matched with dictionary! Please check video filenames.")
            return
        
        # Step 4: Load existing data if resume
        if resume and os.path.exists(config.META_CSV_OUTPUT):
            print("\n[Step 4/5] Loading existing data...")
            self._load_existing_data()
        else:
            print("\n[Step 4/5] Starting fresh (no existing data)")
        
        # Step 5: Extract landmarks and create labels
        print("\n[Step 5/5] Extracting landmarks...")
        self._process_videos(matched_videos)
        
        # Save results
        self._save_results()
        
        # Print final statistics
        self._print_final_statistics()
        
        print("\n" + "=" * 80)
        print("PIPELINE COMPLETED")
        print("=" * 80)
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def _match_videos_with_dictionary(self, videos):
        """Match videos với dictionary entries"""
        matched = []
        
        for video in tqdm(videos, desc="Matching"):
            entry = self.dictionary_manager.match_video_file(video['filename'])
            
            if entry:
                video['matched'] = True
                video['label'] = entry.get('word')
                video['label_normalized'] = entry.get('_word')
                video['description'] = entry.get('description')
                video['type'] = entry.get('tl')
                video['video_id'] = entry.get('_id')
                video['signer'] = self.dictionary_manager.get_signer_from_filename(video['filename'])
                
                matched.append(video)
            else:
                video['matched'] = False
        
        return matched
    
    def _load_existing_data(self):
        """Load existing meta.csv and processed videos"""
        try:
            df = pd.read_csv(config.META_CSV_OUTPUT)
            self.processed_videos = df['orig_name'].tolist()
            print(f"  ✓ Loaded {len(self.processed_videos)} previously processed videos")
            
            if config.SKIP_EXISTING:
                print(f"  ℹ  Will skip these videos")
        except Exception as e:
            print(f"  ⚠️  Could not load existing data: {e}")
            self.processed_videos = []
    
    def _process_videos(self, videos):
        """Process videos: extract landmarks and create labels"""
        total = len(videos)
        skipped = 0
        
        print(f"\nProcessing {total} videos...")
        print("=" * 80)
        
        for i, video in enumerate(tqdm(videos, desc="Extracting")):
            video_name = video['filename']
            
            # Skip if already processed
            if config.SKIP_EXISTING and video_name in self.processed_videos:
                skipped += 1
                continue
            
            # Generate output filename
            output_filename = self._generate_landmark_filename(video)
            output_path = os.path.join(config.LANDMARKS_OUTPUT_DIR, output_filename)
            
            # Skip if output file already exists
            if config.SKIP_EXISTING and os.path.exists(output_path):
                skipped += 1
                continue
            
            # Extract landmarks
            print(f"\n[{i+1}/{total}] Processing: {video_name}")
            landmarks = self.landmark_extractor.extract_from_video(video['full_path'])
            
            if landmarks is not None:
                # Validate
                if self.landmark_extractor.validate_landmarks(landmarks):
                    # Save landmarks
                    self.landmark_extractor.save_landmarks(landmarks, output_path)
                    
                    # Create meta entry
                    meta_entry = {
                        'npy': output_path,
                        'label': video.get('video_id', ''),
                        'label_vi': video.get('label', ''),
                        'orig_name': video_name,
                        'signer': video.get('signer', 'UNKNOWN'),
                        'description': video.get('description', ''),
                        'type': video.get('type', ''),
                        'num_frames': len(landmarks),
                        'video_path': video['relative_path']
                    }
                    
                    self.meta_data.append(meta_entry)
                    
                    # Save checkpoint periodically
                    if len(self.meta_data) % config.BATCH_SIZE == 0:
                        self._save_checkpoint()
                else:
                    print(f"  ✗ Landmarks validation failed")
            else:
                print(f"  ✗ Extraction failed")
        
        if skipped > 0:
            print(f"\n✓ Skipped {skipped} already processed videos")
    
    def _generate_landmark_filename(self, video):
        """Generate unique filename for landmarks"""
        # Use video_id if available
        if 'video_id' in video and video['video_id']:
            base_name = f"{video['video_id']}_{video.get('label', 'unknown')}"
        else:
            # Use hash of full path
            hash_str = hashlib.md5(video['full_path'].encode()).hexdigest()[:8]
            base_name = f"{hash_str}_{video.get('label', 'unknown')}"
        
        # Clean filename
        base_name = base_name.replace(' ', '_').replace('/', '_')
        
        return f"{base_name}.npy"
    
    def _save_checkpoint(self):
        """Save checkpoint"""
        if self.meta_data:
            df = pd.DataFrame(self.meta_data)
            checkpoint_path = config.META_CSV_OUTPUT + '.checkpoint'
            df.to_csv(checkpoint_path, index=False, encoding='utf-8')
            print(f"\n  💾 Checkpoint saved: {len(self.meta_data)} entries")
    
    def _save_results(self):
        """Save final results"""
        print("\n" + "=" * 80)
        print("SAVING RESULTS")
        print("=" * 80)
        
        # Create output directory
        os.makedirs(os.path.dirname(config.META_CSV_OUTPUT), exist_ok=True)
        
        # Combine with existing data if any
        all_meta_data = self.meta_data.copy()
        
        if os.path.exists(config.META_CSV_OUTPUT):
            try:
                existing_df = pd.read_csv(config.META_CSV_OUTPUT)
                existing_data = existing_df.to_dict('records')
                
                # Merge (avoid duplicates)
                existing_names = set(existing_df['orig_name'].tolist())
                for entry in all_meta_data:
                    if entry['orig_name'] not in existing_names:
                        existing_data.append(entry)
                
                all_meta_data = existing_data
                print(f"✓ Merged with {len(existing_df)} existing entries")
            except Exception as e:
                print(f"⚠️  Could not merge with existing data: {e}")
        
        # Save meta.csv
        if all_meta_data:
            df = pd.DataFrame(all_meta_data)
            df.to_csv(config.META_CSV_OUTPUT, index=False, encoding='utf-8')
            print(f"✓ Saved meta.csv: {len(df)} entries")
            print(f"  Path: {config.META_CSV_OUTPUT}")
        else:
            print("⚠️  No data to save")
        
        # Save failed videos log
        self.landmark_extractor.save_failed_videos_log(config.FAILED_VIDEOS_LOG)
        
        # Save statistics
        stats = {
            'timestamp': datetime.now().isoformat(),
            'dictionary_stats': self.dictionary_manager.get_statistics(),
            'video_stats': self.video_scanner.get_statistics(),
            'extraction_stats': self.landmark_extractor.get_statistics(),
            'meta_entries': len(all_meta_data)
        }
        
        with open(config.STATS_OUTPUT, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Saved statistics: {config.STATS_OUTPUT}")
        
        # Remove checkpoint if exists
        checkpoint_path = config.META_CSV_OUTPUT + '.checkpoint'
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            print(f"✓ Removed checkpoint file")
    
    def _print_final_statistics(self):
        """Print final statistics"""
        print("\n" + "=" * 80)
        print("FINAL STATISTICS")
        print("=" * 80)
        
        self.landmark_extractor.print_statistics()
        
        if self.meta_data:
            df = pd.DataFrame(self.meta_data)
            
            print("\nLabel distribution:")
            label_counts = df['label_vi'].value_counts()
            for label, count in label_counts.head(20).items():
                print(f"  {label}: {count}")
            
            if len(label_counts) > 20:
                print(f"  ... and {len(label_counts) - 20} more")
            
            print(f"\nSigner distribution:")
            signer_counts = df['signer'].value_counts()
            for signer, count in signer_counts.items():
                print(f"  {signer}: {count}")
            
            print(f"\nFrames statistics:")
            print(f"  Total frames: {df['num_frames'].sum()}")
            print(f"  Average per video: {df['num_frames'].mean():.1f}")
            print(f"  Min: {df['num_frames'].min()}")
            print(f"  Max: {df['num_frames'].max()}")
        
        print("=" * 80)


def main():
    """Main function"""
    import sys
    
    # Parse arguments
    resume = '--no-resume' not in sys.argv
    
    # Create pipeline
    pipeline = DataLabelingPipeline()
    
    # Run
    try:
        pipeline.run(resume=resume)
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        print("Progress has been saved. Run again with resume to continue.")
    except Exception as e:
        print(f"\n\n✗ Pipeline error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
