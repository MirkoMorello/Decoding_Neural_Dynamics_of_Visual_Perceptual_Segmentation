# Decoding Neural Dynamics of Visual Perceptual Segmentation

This repository contains the complete source code for the Master of Science thesis of the same name. It implements and evaluates a novel architecture for computational gaze prediction, based on the hypothesis that explicitly leveraging visual segmentation information can significantly improve performance.

The final model, **DinoGaze-SPADE**, sets a new state-of-the-art on standard saliency and scanpath prediction benchmarks.

## Getting Started

Follow these steps to set up the environment and run the training pipeline.

### 1. Installation

First, install all the required dependencies using the `setup` task defined in the `pixi.toml` file. This will install everything from `conda` and `pip`, including the correct GPU-enabled PyTorch version.

```bash
pixi run setup
```

### 2. Running the Training Pipeline

Once the setup is complete, you can run the main orchestrator script. This script executes the training stages defined in the master configuration file (`configs/v2/pipeline.yaml`).

You need to specify the number of GPUs you want to use for the training.

```bash
pixi run python -m src.orchestrator --master-config configs/v2/pipeline.yaml --nproc_per_node=<num_gpus>
```

Replace `<num_gpus>` with the number of GPUs you have available (e.g., `2`).

## Project Overview

This project is built on the theory that the segmentation of a visual scene is not just a preliminary step for recognition, but an integral part of the mechanism of gaze itself. This codebase tests that hypothesis by building and evaluating models that directly use segmentation information to predict where humans look.

### Key Architectural Concepts

The codebase is designed to be modular, reproducible, and scalable. The key components include:

*   **DinoGaze-SPADE Model (`src/models/`)**: The final and best-performing model. Its architecture is defined by two main innovations:
    1.  **A DINOv2 Vision Transformer Backbone**: Used as a powerful feature extractor, providing a rich, multi-scale representation of the input image.
    2.  **Dynamic SPADE for Information Injection**: Instead of relying on the network to learn about segments implicitly, we inject segmentation information directly into the model. This is achieved via a "semantic painting" technique, where a feature-rich map of the scene's segments is used to modulate the network's activations using Spatially-Adaptive Normalization (SPADE) layers.

*   **Experiment Orchestrator (`src/orchestrator.py`)**: A top-level script that manages complex, multi-stage experiments (e.g., pre-training then fine-tuning). It reads a master `pipeline.yaml` file to run a sequence of training stages, handling configuration and checkpointing between them.

*   **Generic Training Engine (`src/training.py`)**: A reusable and robust training loop that handles the complexities of training modern deep learning models. It features support for:
    -   Distributed Data Parallel (DDP) for multi-GPU training.
    -   Automatic Mixed Precision (AMP) for faster training and reduced memory usage.
    -   Gradient Accumulation to simulate large batch sizes.

*   **Efficient Data Pipeline (`src/data.py`, `src/datasets/`)**: The data loading pipeline is highly optimized for performance:
    -   **LMDB Caching**: Pre-processed datasets are cached in a Lightning Memory-Mapped Database (LMDB) for extremely fast read access, eliminating I/O bottlenecks.
    -   **Shape-Aware Batching**: A custom `ImageDatasetSampler` groups images of similar sizes into batches, minimizing padding and maximizing GPU memory efficiency.

*   **Component Registry (`src/registry.py`)**: A flexible registry system that uses decorators (`@register_model`, `@register_data`). This allows new models and datasets to be added to the project just by creating a new file, without needing to modify the core training or orchestration code.
