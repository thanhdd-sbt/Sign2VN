"""
Landmark Extractor - Extract landmarks từ videos using MediaPipe
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, Tuple
import os

import labeling_config as config


class LandmarkExtractor:
    """Extract landmarks từ videos using MediaPipe Holistic"""
    
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=config.MEDIAPIPE_MODEL_COMPLEXITY,
            min_detection_confidence=config.MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MEDIAPIPE_MIN_TRACKING_CONFIDENCE
        )
        
        self.stats = {
            'total_videos': 0,
            'successful': 0,
            'failed': 0,
            'total_frames': 0,
            'failed_videos': []
        }
    
    def extract_from_video(self, video_path: str) -> Optional[np.ndarray]:
        """
        Extract landmarks từ video
        
        Args:
            video_path: Đường dẫn đến video
        
        Returns:
            numpy array (num_frames, num_landmarks * 3) hoặc None nếu lỗi
        """
        self.stats['total_videos'] += 1
        
        # Check if video exists
        if not os.path.exists(video_path):
            print(f"  ✗ Video not found: {video_path}")
            self.stats['failed'] += 1
            self.stats['failed_videos'].append({
                'path': video_path,
                'reason': 'File not found'
            })
            return None
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"  ✗ Cannot open video: {video_path}")
            self.stats['failed'] += 1
            self.stats['failed_videos'].append({
                'path': video_path,
                'reason': 'Cannot open video'
            })
            return None
        
        landmarks_sequence = []
        frame_count = 0
        failed_frames = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Limit frames if configured
                if config.MAX_FRAMES_PER_VIDEO and frame_count >= config.MAX_FRAMES_PER_VIDEO:
                    break
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process with MediaPipe
                results = self.holistic.process(frame_rgb)
                
                # Extract landmarks
                frame_landmarks = self._extract_frame_landmarks(results)
                
                if frame_landmarks is not None:
                    landmarks_sequence.append(frame_landmarks)
                    frame_count += 1
                else:
                    failed_frames += 1
                
        except Exception as e:
            print(f"  ✗ Error processing video: {e}")
            self.stats['failed'] += 1
            self.stats['failed_videos'].append({
                'path': video_path,
                'reason': str(e)
            })
            return None
        finally:
            cap.release()
        
        # Check if we have enough frames
        if frame_count < config.MIN_FRAMES_REQUIRED:
            print(f"  ✗ Not enough valid frames: {frame_count}/{config.MIN_FRAMES_REQUIRED}")
            self.stats['failed'] += 1
            self.stats['failed_videos'].append({
                'path': video_path,
                'reason': f'Only {frame_count} valid frames (min: {config.MIN_FRAMES_REQUIRED})'
            })
            return None
        
        # Convert to numpy array
        landmarks_array = np.array(landmarks_sequence, dtype=np.float32)
        
        self.stats['successful'] += 1
        self.stats['total_frames'] += frame_count
        
        print(f"  ✓ Extracted {frame_count} frames ({failed_frames} failed)")
        
        return landmarks_array
    
    def _extract_frame_landmarks(self, results) -> Optional[np.ndarray]:
        """
        Extract landmarks từ MediaPipe results cho 1 frame
        
        Returns:
            numpy array (num_landmarks * 3,) hoặc None nếu không detect được
        """
        frame_landmarks = []
        has_any_landmarks = False
        
        # Pose landmarks (33 points)
        if results.pose_landmarks:
            has_any_landmarks = True
            for landmark in results.pose_landmarks.landmark:
                frame_landmarks.extend([landmark.x, landmark.y, landmark.z])
        else:
            frame_landmarks.extend([0.0] * (config.NUM_POSE_LANDMARKS * 3))
        
        # Left hand landmarks (21 points)
        if results.left_hand_landmarks:
            has_any_landmarks = True
            for landmark in results.left_hand_landmarks.landmark:
                frame_landmarks.extend([landmark.x, landmark.y, landmark.z])
        else:
            frame_landmarks.extend([0.0] * (config.NUM_HAND_LANDMARKS * 3))
        
        # Right hand landmarks (21 points)
        if results.right_hand_landmarks:
            has_any_landmarks = True
            for landmark in results.right_hand_landmarks.landmark:
                frame_landmarks.extend([landmark.x, landmark.y, landmark.z])
        else:
            frame_landmarks.extend([0.0] * (config.NUM_HAND_LANDMARKS * 3))
        
        # Face landmarks (468 points)
        if results.face_landmarks:
            has_any_landmarks = True
            for landmark in results.face_landmarks.landmark:
                frame_landmarks.extend([landmark.x, landmark.y, landmark.z])
        else:
            frame_landmarks.extend([0.0] * (config.NUM_FACE_LANDMARKS * 3))
        
        # Return None if no landmarks detected at all
        if not has_any_landmarks:
            return None
        
        return np.array(frame_landmarks, dtype=np.float32)
    
    def validate_landmarks(self, landmarks: np.ndarray) -> bool:
        """
        Validate landmarks quality
        
        Args:
            landmarks: numpy array (num_frames, num_landmarks * 3)
        
        Returns:
            True if valid, False otherwise
        """
        if landmarks is None or len(landmarks) == 0:
            return False
        
        # Check shape
        expected_features = config.TOTAL_LANDMARKS * config.LANDMARK_DIM
        if landmarks.shape[1] != expected_features:
            return False
        
        # Check for too many zeros (indicates poor detection)
        zero_ratio = np.sum(landmarks == 0) / landmarks.size
        if zero_ratio > 0.8:  # More than 80% zeros
            return False
        
        return True
    
    def save_landmarks(self, landmarks: np.ndarray, output_path: str):
        """
        Save landmarks to .npy file
        
        Args:
            landmarks: numpy array
            output_path: Path to save file
        """
        # Create directory if not exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save
        np.save(output_path, landmarks)
    
    def get_statistics(self):
        """Get extraction statistics"""
        return self.stats
    
    def print_statistics(self):
        """Print extraction statistics"""
        stats = self.stats
        
        print("\n" + "=" * 80)
        print("EXTRACTION STATISTICS")
        print("=" * 80)
        print(f"Total videos processed: {stats['total_videos']}")
        print(f"Successful: {stats['successful']}")
        print(f"Failed: {stats['failed']}")
        
        if stats['successful'] > 0:
            print(f"Total frames extracted: {stats['total_frames']}")
            print(f"Average frames per video: {stats['total_frames']/stats['successful']:.1f}")
        
        success_rate = (stats['successful'] / stats['total_videos'] * 100) if stats['total_videos'] > 0 else 0
        print(f"Success rate: {success_rate:.1f}%")
        
        if stats['failed_videos']:
            print(f"\nFailed videos: {len(stats['failed_videos'])}")
            print("(See failed_videos.txt for details)")
        
        print("=" * 80)
    
    def save_failed_videos_log(self, output_path: str):
        """Save failed videos log"""
        if not self.stats['failed_videos']:
            return
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("FAILED VIDEOS LOG\n")
            f.write("=" * 80 + "\n\n")
            
            for i, failed in enumerate(self.stats['failed_videos'], 1):
                f.write(f"{i}. {failed['path']}\n")
                f.write(f"   Reason: {failed['reason']}\n\n")
        
        print(f"\n✓ Failed videos log saved to: {output_path}")
    
    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'holistic'):
            self.holistic.close()


def test_landmark_extractor():
    """Test LandmarkExtractor with a sample video"""
    print("Testing LandmarkExtractor...")
    
    # You need to provide a test video path
    test_video = "/path/to/test/video.mp4"
    
    if not os.path.exists(test_video):
        print(f"Test video not found: {test_video}")
        print("Please update test_video path in test_landmark_extractor()")
        return
    
    extractor = LandmarkExtractor()
    
    print(f"\nExtracting landmarks from: {test_video}")
    landmarks = extractor.extract_from_video(test_video)
    
    if landmarks is not None:
        print(f"\n✓ Extraction successful!")
        print(f"  Shape: {landmarks.shape}")
        print(f"  Dtype: {landmarks.dtype}")
        print(f"  Min: {landmarks.min():.4f}")
        print(f"  Max: {landmarks.max():.4f}")
        print(f"  Mean: {landmarks.mean():.4f}")
        
        # Validate
        is_valid = extractor.validate_landmarks(landmarks)
        print(f"  Valid: {is_valid}")
        
        # Save test
        test_output = "/tmp/test_landmarks.npy"
        extractor.save_landmarks(landmarks, test_output)
        print(f"\n✓ Saved to: {test_output}")
    else:
        print(f"\n✗ Extraction failed")
    
    extractor.print_statistics()


if __name__ == "__main__":
    test_landmark_extractor()
