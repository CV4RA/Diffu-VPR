
# MambaRainPlace

This repository contains the implementation of the paper "Place Recognizer Meet Rainy Day: Towards Robust Visual Place Recognition for Mobile Robots with an End-to-end Enhanced Net". The code is structured similarly to the [MambaPlace](https://github.com/CV4RA/MambaPlace) repository and provides a complete end-to-end pipeline for visual place recognition under rainy conditions.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Testing](#testing)
- [Configuration](#configuration)
- [Dataset](#dataset)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

diffuPlace is a robust visual place recognition (VPR) system designed for mobile robots to perform place recognition under adverse weather conditions, particularly rainy scenes. The system integrates a multi-scale sampling network (MSSN) for rain removal and a multi-stage feature pyramid network (MSFPN) for enhanced feature extraction.

Key components:
- **MSSN:** A lightweight network for removing rain effects from images.
- **MSFPN:** A pyramid transformer-based network for extracting multi-scale features and aggregating them into global descriptors.
- **Matcher:** A triplet-based place matcher that handles challenging VPR tasks by effectively distinguishing between similar and dissimilar places.

## Installation

### Requirements

Ensure you have the following dependencies installed:

- Python 3.8+
- PyTorch 1.9.0+
- torchvision
- numpy
- pyyaml
- PIL

You can install the required packages using:

```bash
pip install -r requirements.txt
```

### Clone the Repository

```bash
git clone https://github.com/yourusername/MambaRainPlace.git
cd MambaRainPlace
```

## Usage

### Training

To train the model on your dataset, use the following command:

```bash
python train.py --config config/default.yaml
```

This will start the training process using the parameters defined in the `config/default.yaml` file.

### Testing

After training, you can evaluate the model's performance by running:

```bash
python test.py --config config/default.yaml --checkpoint path/to/checkpoint.pth
```

## Configuration

The training and model parameters are defined in YAML configuration files. You can adjust settings such as the learning rate, batch size, and model architecture by editing the configuration file located at `config/default.yaml`.

Example configuration:

```yaml
model:
  name: RainyVPRNet
  MSSN:
    channels: 16
    num_dense_blocks: 5
  MSFPN:
    vlad_clusters: 64
    dim: 128
training:
  optimizer: AdamW
  learning_rate: 0.0005
  batch_size: 16
  epochs: 50
```

## Dataset

The dataset required for training and testing should be organized as follows:

```
dataset/
├── train/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── test/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

Modify the dataset paths in the configuration file to point to your dataset directory.

## Results

The model is evaluated on standard VPR benchmarks such as RainPlace and RainSim. Precision-Recall and Recall@N metrics are used to assess performance. Example results will be provided after running the testing script.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an issue if you find any bugs or have suggestions for improvements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
