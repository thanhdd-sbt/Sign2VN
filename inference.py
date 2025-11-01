"""
Inference script - Dự đoán ngôn ngữ ký hiệu từ video mới
"""

import os
import torch
import numpy as np
import pickle
import argparse
from typing import List, Tuple
import cv2
import mediapipe as mp

import config
from model import Sign2TextModel
from data_loader import VietnameseTokenizer


class SignLanguagePredictor:
    """Class để dự đoán ngôn ngữ ký hiệu từ video"""
    
    def __init__(self, checkpoint_path: str, tokenizer_path: str):
        """
        Args:
            checkpoint_path: Đường dẫn đến model checkpoint
            tokenizer_path: Đường dẫn đến tokenizer
        """
        print("Loading model and tokenizer...")
        
        # Load tokenizer
        with open(tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)
        
        # Load model
        self.model = Sign2TextModel(vocab_size=self.tokenizer.vocab_size)
        checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(config.DEVICE)
        self.model.eval()
        
        print(f"Model loaded from {checkpoint_path}")
        print(f"Tokenizer loaded with vocab size: {self.tokenizer.vocab_size}")
        
        # Initialize MediaPipe
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def extract_landmarks_from_video(self, video_path: str) -> np.ndarray:
        """
        Trích xuất landmarks từ video sử dụng MediaPipe
        
        Args:
            video_path: Đường dẫn đến video
        
        Returns:
            landmarks: numpy array shape (num_frames, num_landmarks * 3)
        """
        print(f"Extracting landmarks from {video_path}...")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        landmarks_sequence = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.holistic.process(frame_rgb)
            
            # Extract landmarks
            frame_landmarks = []
            
            # Pose landmarks (33 points)
            if results.pose_landmarks:
                for landmark in results.pose_landmarks.landmark:
                    frame_landmarks.extend([landmark.x, landmark.y, landmark.z])
            else:
                frame_landmarks.extend([0.0] * (config.NUM_POSE_LANDMARKS * 3))
            
            # Left hand landmarks (21 points)
            if results.left_hand_landmarks:
                for landmark in results.left_hand_landmarks.landmark:
                    frame_landmarks.extend([landmark.x, landmark.y, landmark.z])
            else:
                frame_landmarks.extend([0.0] * (config.NUM_HAND_LANDMARKS * 3))
            
            # Right hand landmarks (21 points)
            if results.right_hand_landmarks:
                for landmark in results.right_hand_landmarks.landmark:
                    frame_landmarks.extend([landmark.x, landmark.y, landmark.z])
            else:
                frame_landmarks.extend([0.0] * (config.NUM_HAND_LANDMARKS * 3))
            
            # Face landmarks (468 points)
            if results.face_landmarks:
                for landmark in results.face_landmarks.landmark:
                    frame_landmarks.extend([landmark.x, landmark.y, landmark.z])
            else:
                frame_landmarks.extend([0.0] * (config.NUM_FACE_LANDMARKS * 3))
            
            landmarks_sequence.append(frame_landmarks)
            frame_count += 1
        
        cap.release()
        
        if frame_count == 0:
            raise ValueError(f"No frames extracted from video: {video_path}")
        
        landmarks_array = np.array(landmarks_sequence)
        print(f"Extracted {frame_count} frames with shape {landmarks_array.shape}")
        
        return landmarks_array
    
    def extract_landmarks_from_npy(self, npy_path: str) -> np.ndarray:
        """
        Load landmarks từ file .npy đã có sẵn
        
        Args:
            npy_path: Đường dẫn đến file .npy
        
        Returns:
            landmarks: numpy array
        """
        print(f"Loading landmarks from {npy_path}...")
        landmarks = np.load(npy_path)
        print(f"Loaded landmarks with shape {landmarks.shape}")
        return landmarks
    
    def preprocess_landmarks(self, landmarks: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Preprocess landmarks cho model
        
        Args:
            landmarks: numpy array (num_frames, num_features)
        
        Returns:
            landmarks_tensor: (1, max_seq_len, num_features)
            length: (1,) actual length
        """
        # Normalize
        mean = np.mean(landmarks, axis=0)
        std = np.std(landmarks, axis=0)
        std = np.where(std == 0, 1, std)
        normalized = (landmarks - mean) / std
        
        # Get actual length
        actual_length = min(len(normalized), config.MAX_SEQUENCE_LENGTH)
        
        # Pad or truncate
        if len(normalized) > config.MAX_SEQUENCE_LENGTH:
            normalized = normalized[:config.MAX_SEQUENCE_LENGTH]
        elif len(normalized) < config.MAX_SEQUENCE_LENGTH:
            pad_length = config.MAX_SEQUENCE_LENGTH - len(normalized)
            padding = np.zeros((pad_length, normalized.shape[1]))
            normalized = np.vstack([normalized, padding])
        
        # Convert to tensor
        landmarks_tensor = torch.FloatTensor(normalized).unsqueeze(0)  # Add batch dimension
        length = torch.LongTensor([actual_length])
        
        return landmarks_tensor, length
    
    def predict(
        self,
        landmarks: np.ndarray,
        max_length: int = 50,
        return_attention: bool = False
    ) -> dict:
        """
        Dự đoán text từ landmarks
        
        Args:
            landmarks: numpy array (num_frames, num_features)
            max_length: Độ dài tối đa của output
            return_attention: Có return attention weights không
        
        Returns:
            dict chứa prediction và các thông tin khác
        """
        # Preprocess
        landmarks_tensor, length = self.preprocess_landmarks(landmarks)
        landmarks_tensor = landmarks_tensor.to(config.DEVICE)
        length = length.to(config.DEVICE)
        
        # Generate
        with torch.no_grad():
            generated_tokens, attention_weights = self.model.generate(
                landmarks_tensor,
                length,
                max_length=max_length,
                sos_token=self.tokenizer.word2idx[config.SOS_TOKEN],
                eos_token=self.tokenizer.word2idx[config.EOS_TOKEN]
            )
        
        # Decode to text
        predicted_text = self.tokenizer.decode(
            generated_tokens[0].cpu().tolist(),
            skip_special=True
        )
        
        result = {
            'text': predicted_text,
            'tokens': generated_tokens[0].cpu().tolist(),
            'num_frames': length.item()
        }
        
        if return_attention:
            result['attention_weights'] = [att.cpu().numpy() for att in attention_weights]
        
        return result
    
    def predict_from_video(
        self,
        video_path: str,
        max_length: int = 50,
        save_npy: bool = False
    ) -> dict:
        """
        Dự đoán từ video file
        
        Args:
            video_path: Đường dẫn đến video
            max_length: Độ dài tối đa của output
            save_npy: Có lưu landmarks thành file .npy không
        
        Returns:
            dict chứa prediction
        """
        # Extract landmarks
        landmarks = self.extract_landmarks_from_video(video_path)
        
        # Save landmarks if requested
        if save_npy:
            npy_path = video_path.replace('.mp4', '_landmarks.npy')
            np.save(npy_path, landmarks)
            print(f"Landmarks saved to {npy_path}")
        
        # Predict
        result = self.predict(landmarks, max_length)
        result['video_path'] = video_path
        
        return result
    
    def predict_from_npy(
        self,
        npy_path: str,
        max_length: int = 50
    ) -> dict:
        """
        Dự đoán từ file .npy
        
        Args:
            npy_path: Đường dẫn đến file .npy
            max_length: Độ dài tối đa của output
        
        Returns:
            dict chứa prediction
        """
        # Load landmarks
        landmarks = self.extract_landmarks_from_npy(npy_path)
        
        # Predict
        result = self.predict(landmarks, max_length)
        result['npy_path'] = npy_path
        
        return result
    
    def batch_predict_from_folder(
        self,
        folder_path: str,
        file_extension: str = '.npy',
        max_length: int = 50
    ) -> List[dict]:
        """
        Dự đoán cho nhiều files trong folder
        
        Args:
            folder_path: Đường dẫn đến folder
            file_extension: Extension của files (.npy hoặc .mp4)
            max_length: Độ dài tối đa của output
        
        Returns:
            List of predictions
        """
        print(f"\nBatch prediction from {folder_path}")
        print(f"Looking for {file_extension} files...")
        
        # Get all files
        files = [f for f in os.listdir(folder_path) if f.endswith(file_extension)]
        print(f"Found {len(files)} files")
        
        results = []
        for i, filename in enumerate(files):
            print(f"\nProcessing {i+1}/{len(files)}: {filename}")
            
            file_path = os.path.join(folder_path, filename)
            
            try:
                if file_extension == '.npy':
                    result = self.predict_from_npy(file_path, max_length)
                elif file_extension == '.mp4':
                    result = self.predict_from_video(file_path, max_length)
                else:
                    print(f"Unsupported file extension: {file_extension}")
                    continue
                
                result['filename'] = filename
                results.append(result)
                
                print(f"Prediction: {result['text']}")
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue
        
        return results


def main(args):
    """Main function cho inference"""
    
    # Initialize predictor
    predictor = SignLanguagePredictor(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer
    )
    
    print("\n" + "=" * 80)
    print("SIGN LANGUAGE TO VIETNAMESE TEXT - INFERENCE")
    print("=" * 80)
    
    # Single file prediction
    if args.video_path:
        print("\n[Single Video Prediction]")
        result = predictor.predict_from_video(
            args.video_path,
            max_length=args.max_length,
            save_npy=args.save_npy
        )
        print(f"\nVideo: {result['video_path']}")
        print(f"Prediction: {result['text']}")
        print(f"Number of frames: {result['num_frames']}")
    
    elif args.npy_path:
        print("\n[Single NPY Prediction]")
        result = predictor.predict_from_npy(
            args.npy_path,
            max_length=args.max_length
        )
        print(f"\nNPY file: {result['npy_path']}")
        print(f"Prediction: {result['text']}")
        print(f"Number of frames: {result['num_frames']}")
    
    # Batch prediction
    elif args.folder_path:
        print("\n[Batch Prediction]")
        results = predictor.batch_predict_from_folder(
            args.folder_path,
            file_extension=args.file_extension,
            max_length=args.max_length
        )
        
        print("\n" + "=" * 80)
        print(f"BATCH PREDICTION RESULTS ({len(results)} files)")
        print("=" * 80)
        
        for i, result in enumerate(results):
            print(f"\n{i+1}. {result['filename']}")
            print(f"   Prediction: {result['text']}")
        
        # Save results
        if args.output:
            import json
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\nResults saved to {args.output}")
    
    else:
        print("Please provide --video_path, --npy_path, or --folder_path")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sign Language Inference")
    
    # Model arguments
    parser.add_argument('--checkpoint', type=str,
                        default=f"{config.CHECKPOINT_DIR}/best_model.pt",
                        help='Path to model checkpoint')
    parser.add_argument('--tokenizer', type=str,
                        default=f"{config.CHECKPOINT_DIR}/tokenizer.pkl",
                        help='Path to tokenizer')
    
    # Input arguments
    parser.add_argument('--video_path', type=str, default=None,
                        help='Path to single video file')
    parser.add_argument('--npy_path', type=str, default=None,
                        help='Path to single .npy file')
    parser.add_argument('--folder_path', type=str, default=None,
                        help='Path to folder for batch prediction')
    parser.add_argument('--file_extension', type=str, default='.npy',
                        choices=['.npy', '.mp4'],
                        help='File extension for batch prediction')
    
    # Output arguments
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file for batch predictions')
    parser.add_argument('--save_npy', action='store_true',
                        help='Save extracted landmarks as .npy')
    
    # Generation arguments
    parser.add_argument('--max_length', type=int, default=50,
                        help='Maximum generation length')
    
    args = parser.parse_args()
    
    main(args)
