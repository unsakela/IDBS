"""
Temporal Convolutional Network (TCN) for IDBS System

This module implements a TCN with dilated causal convolutions for processing
time-series sensor data in the Intelligent Driver Behavior Sensing system.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TemporalBlock(nn.Module):
    """
    Temporal block with dilated causal convolution, residual connection,
    and optional skip connection.
    """
    
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        
        # Downsampling connection if input and output dimensions differ
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights using Xavier initialization."""
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
    
    def forward(self, x):
        """
        Forward pass through the temporal block.
        
        Args:
            x: Input tensor of shape (batch_size, n_inputs, seq_len)
            
        Returns:
            Output tensor of shape (batch_size, n_outputs, seq_len)
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class Chomp1d(nn.Module):
    """
    Module to remove padding from the end of the sequence to maintain causality.
    """
    
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    
    def forward(self, x):
        """
        Remove the last chomp_size elements from the sequence.
        
        Args:
            x: Input tensor of shape (batch_size, channels, seq_len)
            
        Returns:
            Tensor with padding removed
        """
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network (TCN) with multiple temporal blocks.
    """
    
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers += [
                TemporalBlock(
                    in_channels, out_channels, kernel_size,
                    stride=1, dilation=dilation_size,
                    padding=(kernel_size-1) * dilation_size, dropout=dropout
                )
            ]
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass through the TCN.
        
        Args:
            x: Input tensor of shape (batch_size, num_inputs, seq_len)
            
        Returns:
            Output tensor of shape (batch_size, num_channels[-1], seq_len)
        """
        return self.network(x)


class TCN(nn.Module):
    """
    Complete TCN model for time-series sensor data processing.
    
    This model processes multivariate time-series sensor data using dilated
    causal convolutions to capture long-range temporal dependencies while
    maintaining causality.
    """
    
    def __init__(self, input_size, num_channels, kernel_size=3, dropout=0.2, 
                 output_size=None, return_sequences=True):
        super(TCN, self).__init__()
        
        self.input_size = input_size
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.output_size = output_size
        self.return_sequences = return_sequences
        
        # Temporal Convolutional Network
        self.tcn = TemporalConvNet(
            num_inputs=input_size,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout
        )
        
        # Output projection layer
        if output_size is not None:
            self.output_projection = nn.Linear(num_channels[-1], output_size)
        else:
            self.output_projection = None
    
    def forward(self, x):
        """
        Forward pass through the TCN.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size) or
               (batch_size, input_size, seq_len)
               
        Returns:
            Output tensor. Shape depends on return_sequences and output_size:
            - If return_sequences=True and output_size=None: (batch_size, seq_len, num_channels[-1])
            - If return_sequences=False and output_size=None: (batch_size, num_channels[-1])
            - If return_sequences=True and output_size is not None: (batch_size, seq_len, output_size)
            - If return_sequences=False and output_size is not None: (batch_size, output_size)
        """
        # Ensure input is in (batch_size, input_size, seq_len) format
        if x.dim() == 3 and x.size(1) != self.input_size:
            x = x.transpose(1, 2)  # (batch_size, seq_len, input_size) -> (batch_size, input_size, seq_len)
        
        # Pass through TCN
        tcn_output = self.tcn(x)  # (batch_size, num_channels[-1], seq_len)
        
        # Transpose back to (batch_size, seq_len, num_channels[-1])
        tcn_output = tcn_output.transpose(1, 2)
        
        # Apply output projection if specified
        if self.output_projection is not None:
            tcn_output = self.output_projection(tcn_output)
        
        # Return sequences or final output only
        if not self.return_sequences:
            tcn_output = tcn_output[:, -1, :]  # Take last time step
        
        return tcn_output
    
    def get_receptive_field(self):
        """
        Calculate the receptive field size of the TCN.
        
        Returns:
            Receptive field size (number of time steps)
        """
        receptive_field = 1
        for i in range(len(self.num_channels)):
            receptive_field += (self.kernel_size - 1) * (2 ** i)
        return receptive_field


if __name__ == "__main__":
    """Test the TCN module with sample sensor data."""
    print("Testing Temporal Convolutional Network...")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Model parameters
    batch_size = 8
    seq_len = 100
    input_size = 6  # 6 sensor channels (e.g., 3-axis accelerometer + 3-axis gyroscope)
    num_channels = [64, 128, 256]  # Channel sizes for each layer
    kernel_size = 3
    dropout = 0.2
    
    # Create sample input tensor (simulating sensor data)
    sensor_data = torch.randn(batch_size, seq_len, input_size)
    
    print(f"Input sensor data shape: {sensor_data.shape}")
    print(f"Input size: {input_size}, Sequence length: {seq_len}")
    
    # Initialize TCN model
    tcn_model = TCN(
        input_size=input_size,
        num_channels=num_channels,
        kernel_size=kernel_size,
        dropout=dropout,
        return_sequences=True
    )
    
    print(f"\nTCN Model initialized:")
    print(f"- Number of layers: {len(num_channels)}")
    print(f"- Channel sizes: {num_channels}")
    print(f"- Kernel size: {kernel_size}")
    print(f"- Receptive field: {tcn_model.get_receptive_field()} time steps")
    print(f"- Total parameters: {sum(p.numel() for p in tcn_model.parameters()):,}")
    
    # Forward pass with sequence output
    output_sequences = tcn_model(sensor_data)
    print(f"\nOutput sequences shape: {output_sequences.shape}")
    
    # Test with final output only
    tcn_model_final = TCN(
        input_size=input_size,
        num_channels=num_channels,
        kernel_size=kernel_size,
        dropout=dropout,
        return_sequences=False
    )
    
    output_final = tcn_model_final(sensor_data)
    print(f"Final output shape: {output_final.shape}")
    
    # Test with output projection
    tcn_model_proj = TCN(
        input_size=input_size,
        num_channels=num_channels,
        kernel_size=kernel_size,
        dropout=dropout,
        output_size=128,
        return_sequences=True
    )
    
    output_proj = tcn_model_proj(sensor_data)
    print(f"Projected output shape: {output_proj.shape}")
    
    # Test with different input format
    sensor_data_alt = torch.randn(batch_size, input_size, seq_len)
    output_alt = tcn_model(sensor_data_alt)
    print(f"\nOutput with alternative input format: {output_alt.shape}")
    
    print("\n✓ TCN Module test completed successfully!")
