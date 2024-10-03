Introduction
This project implements a deep learning architecture to tackle the problem of speech denoising and separation in challenging environments. The model processes noisy speech signals and isolates clean speech by filtering out background noise.

The proposed method extends ConvTasNet by introducing key innovations:

Layer normalization techniques (Channel-wise and Global normalization)
Depthwise separable convolutions for efficient computation
Channel shuffle mechanism for better feature extraction
ODE layers to capture temporal dynamics in speech signals
Features
Robust to various noisy environments.
End-to-end model for real-time denoising.
ODE-based architecture for modeling complex temporal dynamics.
Supports large-scale datasets for training.
Si-SNR loss function for high-quality speech enhancement.
Architecture Overview
The architecture includes the following components:

Encoder: Transforms the raw audio waveform into a feature space using 1D convolution.
Normalization: Channel-wise and global normalization techniques stabilize training.
Convolutional Blocks: Feature extraction using depthwise separable convolutions, channel shuffle, and skip connections.
ODE-Based Block: Models the temporal evolution of speech signals using Ordinary Differential Equations (ODEs).
Mask Estimation: Predicts masks to filter out noise from the encoded features.
Decoder: Reconstructs the denoised waveform from the masked features.

