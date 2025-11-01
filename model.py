"""
Model architecture for Sign Language Translation
CNN + LSTM Encoder + Attention-based Seq2Seq Decoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

import config


class SpatialEncoder(nn.Module):
    """CNN để extract spatial features từ landmarks"""
    
    def __init__(
        self,
        input_dim: int = config.TOTAL_LANDMARKS * config.LANDMARK_DIM,
        filters: list = config.CNN_FILTERS,
        kernel_size: int = config.CNN_KERNEL_SIZE
    ):
        super(SpatialEncoder, self).__init__()
        
        self.conv_layers = nn.ModuleList()
        in_channels = 1
        
        for out_channels in filters:
            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(config.DROPOUT_RATE)
            ))
            in_channels = out_channels
        
        self.output_dim = filters[-1]
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_dim)
        Returns:
            (batch_size, seq_len, output_dim)
        """
        # Reshape cho Conv1d: (batch, channels, seq_len)
        x = x.unsqueeze(1)  # (batch, 1, seq_len, input_dim)
        batch_size, _, seq_len, input_dim = x.shape
        x = x.view(batch_size, 1, seq_len * input_dim)
        
        # Apply conv layers
        for conv in self.conv_layers:
            x = conv(x)
        
        # Reshape back: (batch, channels, seq_len * input_dim) -> (batch, seq_len, channels)
        x = x.view(batch_size, self.output_dim, seq_len, -1)
        x = torch.mean(x, dim=-1)  # Average pool over landmarks dimension
        x = x.transpose(1, 2)  # (batch, seq_len, channels)
        
        return x


class TemporalEncoder(nn.Module):
    """LSTM để capture temporal dependencies"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = config.LSTM_UNITS,
        num_layers: int = config.LSTM_LAYERS,
        dropout: float = config.DROPOUT_RATE
    ):
        super(TemporalEncoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, lengths):
        """
        Args:
            x: (batch_size, seq_len, input_dim)
            lengths: (batch_size,) actual sequence lengths
        Returns:
            outputs: (batch_size, seq_len, hidden_dim*2)
            hidden: tuple of (h_n, c_n)
        """
        # Pack padded sequence
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # LSTM forward
        packed_output, hidden = self.lstm(packed)
        
        # Unpack
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        # Apply dropout
        output = self.dropout(output)
        
        return output, hidden


class Attention(nn.Module):
    """Bahdanau Attention mechanism"""
    
    def __init__(
        self,
        encoder_hidden_dim: int,
        decoder_hidden_dim: int,
        attention_dim: int = config.ATTENTION_DIM
    ):
        super(Attention, self).__init__()
        
        self.encoder_att = nn.Linear(encoder_hidden_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_hidden_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, encoder_outputs, decoder_hidden, mask=None):
        """
        Args:
            encoder_outputs: (batch_size, seq_len, encoder_hidden_dim)
            decoder_hidden: (batch_size, decoder_hidden_dim)
            mask: (batch_size, seq_len) - 1 for valid positions, 0 for padding
        Returns:
            attention_weights: (batch_size, seq_len)
            context: (batch_size, encoder_hidden_dim)
        """
        # encoder_out: (batch, seq_len, att_dim)
        encoder_out = self.encoder_att(encoder_outputs)
        
        # decoder_out: (batch, 1, att_dim)
        decoder_out = self.decoder_att(decoder_hidden).unsqueeze(1)
        
        # Compute attention scores
        # att: (batch, seq_len, 1)
        att = self.full_att(torch.tanh(encoder_out + decoder_out))
        att = att.squeeze(2)  # (batch, seq_len)
        
        # Apply mask if provided
        if mask is not None:
            att = att.masked_fill(mask == 0, -1e10)
        
        # Softmax
        attention_weights = self.softmax(att)
        
        # Context vector: (batch, encoder_hidden_dim)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        context = context.squeeze(1)
        
        return attention_weights, context


class Decoder(nn.Module):
    """Seq2Seq Decoder với Attention"""
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = config.EMBEDDING_DIM,
        encoder_hidden_dim: int = config.ENCODER_HIDDEN_DIM * 2,  # *2 vì bidirectional
        decoder_hidden_dim: int = config.DECODER_HIDDEN_DIM,
        attention_dim: int = config.ATTENTION_DIM,
        dropout: float = config.DROPOUT_RATE
    ):
        super(Decoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.decoder_hidden_dim = decoder_hidden_dim
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.attention = Attention(encoder_hidden_dim, decoder_hidden_dim, attention_dim)
        
        self.lstm = nn.LSTM(
            embedding_dim + encoder_hidden_dim,
            decoder_hidden_dim,
            batch_first=True
        )
        
        self.fc = nn.Linear(decoder_hidden_dim + encoder_hidden_dim + embedding_dim, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input_token, hidden, encoder_outputs, mask=None):
        """
        Single step decoding
        Args:
            input_token: (batch_size,) current input token
            hidden: (h, c) previous LSTM hidden state
            encoder_outputs: (batch_size, seq_len, encoder_hidden_dim)
            mask: (batch_size, seq_len)
        Returns:
            output: (batch_size, vocab_size) logits
            hidden: (h, c) updated hidden state
            attention_weights: (batch_size, seq_len)
        """
        # Embedding: (batch, 1, embedding_dim)
        embedded = self.embedding(input_token).unsqueeze(1)
        embedded = self.dropout(embedded)
        
        # Attention
        h_prev = hidden[0].squeeze(0)  # (batch, decoder_hidden_dim)
        attention_weights, context = self.attention(encoder_outputs, h_prev, mask)
        
        # Concatenate embedding and context
        # lstm_input: (batch, 1, embedding_dim + encoder_hidden_dim)
        lstm_input = torch.cat([embedded, context.unsqueeze(1)], dim=2)
        
        # LSTM step
        output, hidden = self.lstm(lstm_input, hidden)
        output = output.squeeze(1)  # (batch, decoder_hidden_dim)
        
        # Final prediction
        # Concatenate LSTM output, context, and embedding
        prediction = torch.cat([output, context, embedded.squeeze(1)], dim=1)
        prediction = self.fc(prediction)
        
        return prediction, hidden, attention_weights


class Sign2TextModel(nn.Module):
    """Complete Sign Language to Text model"""
    
    def __init__(
        self,
        vocab_size: int,
        input_dim: int = config.TOTAL_LANDMARKS * config.LANDMARK_DIM
    ):
        super(Sign2TextModel, self).__init__()
        
        # Spatial encoder (CNN)
        self.spatial_encoder = SpatialEncoder(input_dim)
        
        # Temporal encoder (LSTM)
        self.temporal_encoder = TemporalEncoder(
            input_dim=self.spatial_encoder.output_dim,
            hidden_dim=config.ENCODER_HIDDEN_DIM
        )
        
        # Decoder
        self.decoder = Decoder(
            vocab_size=vocab_size,
            encoder_hidden_dim=config.ENCODER_HIDDEN_DIM * 2  # bidirectional
        )
        
        self.vocab_size = vocab_size
    
    def encode(self, landmarks, lengths):
        """
        Encode sign language sequence
        Args:
            landmarks: (batch_size, seq_len, input_dim)
            lengths: (batch_size,) actual lengths
        Returns:
            encoder_outputs: (batch_size, seq_len, hidden_dim*2)
            encoder_hidden: tuple of (h_n, c_n)
        """
        # Spatial features
        spatial_features = self.spatial_encoder(landmarks)
        
        # Temporal features
        encoder_outputs, encoder_hidden = self.temporal_encoder(spatial_features, lengths)
        
        return encoder_outputs, encoder_hidden
    
    def decode_step(self, input_token, hidden, encoder_outputs, mask=None):
        """Single decoding step"""
        return self.decoder(input_token, hidden, encoder_outputs, mask)
    
    def forward(self, landmarks, lengths, target_tokens, teacher_forcing_ratio=0.5):
        """
        Forward pass with teacher forcing
        Args:
            landmarks: (batch_size, seq_len, input_dim)
            lengths: (batch_size,) actual lengths
            target_tokens: (batch_size, target_seq_len) target sequences
            teacher_forcing_ratio: probability of using teacher forcing
        Returns:
            outputs: (batch_size, target_seq_len, vocab_size)
        """
        batch_size = landmarks.size(0)
        target_seq_len = target_tokens.size(1)
        
        # Encode
        encoder_outputs, encoder_hidden = self.encode(landmarks, lengths)
        
        # Initialize decoder hidden state from encoder
        # encoder_hidden: (num_layers*2, batch, hidden_dim)
        # Take last layer, concatenate forward and backward
        h_n, c_n = encoder_hidden
        h_forward = h_n[-2, :, :]  # (batch, hidden_dim)
        h_backward = h_n[-1, :, :]
        h_0 = torch.cat([h_forward, h_backward], dim=1)  # (batch, hidden_dim*2)
        
        # Project to decoder hidden dim
        h_0 = nn.Linear(
            config.ENCODER_HIDDEN_DIM * 2, 
            config.DECODER_HIDDEN_DIM
        ).to(landmarks.device)(h_0).unsqueeze(0)
        
        c_forward = c_n[-2, :, :]
        c_backward = c_n[-1, :, :]
        c_0 = torch.cat([c_forward, c_backward], dim=1)
        c_0 = nn.Linear(
            config.ENCODER_HIDDEN_DIM * 2,
            config.DECODER_HIDDEN_DIM
        ).to(landmarks.device)(c_0).unsqueeze(0)
        
        decoder_hidden = (h_0, c_0)
        
        # Create mask for encoder outputs (1 for valid, 0 for padding)
        mask = torch.zeros(batch_size, encoder_outputs.size(1)).to(landmarks.device)
        for i, length in enumerate(lengths):
            mask[i, :length] = 1
        
        # Start with <SOS> token
        decoder_input = target_tokens[:, 0]
        
        outputs = []
        
        for t in range(1, target_seq_len):
            # Decode step
            output, decoder_hidden, _ = self.decode_step(
                decoder_input, decoder_hidden, encoder_outputs, mask
            )
            
            outputs.append(output.unsqueeze(1))
            
            # Teacher forcing
            use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio
            if use_teacher_forcing:
                decoder_input = target_tokens[:, t]
            else:
                decoder_input = output.argmax(1)
        
        outputs = torch.cat(outputs, dim=1)  # (batch, target_seq_len-1, vocab_size)
        
        return outputs
    
    def generate(self, landmarks, lengths, max_length=50, sos_token=1, eos_token=2):
        """
        Generate translation (inference)
        Args:
            landmarks: (batch_size, seq_len, input_dim)
            lengths: (batch_size,)
            max_length: maximum decoding length
            sos_token: start of sequence token id
            eos_token: end of sequence token id
        Returns:
            generated_tokens: (batch_size, generated_length)
            attention_weights: list of attention weights for visualization
        """
        batch_size = landmarks.size(0)
        
        # Encode
        encoder_outputs, encoder_hidden = self.encode(landmarks, lengths)
        
        # Initialize decoder
        h_n, c_n = encoder_hidden
        h_forward = h_n[-2, :, :]
        h_backward = h_n[-1, :, :]
        h_0 = torch.cat([h_forward, h_backward], dim=1)
        h_0 = nn.Linear(
            config.ENCODER_HIDDEN_DIM * 2,
            config.DECODER_HIDDEN_DIM
        ).to(landmarks.device)(h_0).unsqueeze(0)
        
        c_forward = c_n[-2, :, :]
        c_backward = c_n[-1, :, :]
        c_0 = torch.cat([c_forward, c_backward], dim=1)
        c_0 = nn.Linear(
            config.ENCODER_HIDDEN_DIM * 2,
            config.DECODER_HIDDEN_DIM
        ).to(landmarks.device)(c_0).unsqueeze(0)
        
        decoder_hidden = (h_0, c_0)
        
        # Create mask
        mask = torch.zeros(batch_size, encoder_outputs.size(1)).to(landmarks.device)
        for i, length in enumerate(lengths):
            mask[i, :length] = 1
        
        # Start with <SOS>
        decoder_input = torch.LongTensor([sos_token] * batch_size).to(landmarks.device)
        
        generated_tokens = []
        attention_weights_list = []
        
        for _ in range(max_length):
            output, decoder_hidden, attention_weights = self.decode_step(
                decoder_input, decoder_hidden, encoder_outputs, mask
            )
            
            # Get predicted token
            predicted = output.argmax(1)
            generated_tokens.append(predicted.unsqueeze(1))
            attention_weights_list.append(attention_weights.unsqueeze(1))
            
            # Check if all sequences have generated <EOS>
            if (predicted == eos_token).all():
                break
            
            decoder_input = predicted
        
        generated_tokens = torch.cat(generated_tokens, dim=1)
        
        return generated_tokens, attention_weights_list


# Test model
if __name__ == "__main__":
    # Test với dummy data
    batch_size = 4
    seq_len = 100
    vocab_size = 1000
    
    model = Sign2TextModel(vocab_size).to(config.DEVICE)
    
    # Dummy input
    landmarks = torch.randn(batch_size, seq_len, config.TOTAL_LANDMARKS * config.LANDMARK_DIM).to(config.DEVICE)
    lengths = torch.LongTensor([100, 80, 60, 40]).to(config.DEVICE)
    target_tokens = torch.randint(0, vocab_size, (batch_size, 20)).to(config.DEVICE)
    
    # Forward pass
    outputs = model(landmarks, lengths, target_tokens, teacher_forcing_ratio=1.0)
    print(f"Output shape: {outputs.shape}")
    
    # Generate
    generated, attention = model.generate(landmarks, lengths, max_length=20)
    print(f"Generated shape: {generated.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
