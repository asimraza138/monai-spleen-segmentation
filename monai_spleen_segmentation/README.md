# MONAI Spleen Segmentation Project

## Project Overview

This project demonstrates a 3D medical image segmentation pipeline using the MONAI framework. The goal is to segment the spleen from CT scans, a common task in medical imaging for diagnosis, treatment planning, and volumetric analysis. This repository provides a complete, self-contained example, from data handling and model definition to training and evaluation, designed to be a strong addition to a medical imaging or machine learning portfolio.

## Features

-   **MONAI Framework**: Leverages the Medical Open Network for AI (MONAI) for robust and efficient medical image processing.
-   **3D UNet Architecture**: Implements a 3D U-Net, a state-of-the-art convolutional neural network for volumetric segmentation.
-   **Medical Segmentation Decathlon (MSD) Dataset**: Utilizes the Task 09 (Spleen) dataset from the MSD challenge, a widely recognized benchmark in medical image analysis [1].
-   **Comprehensive Data Preprocessing**: Includes MONAI transforms for NIfTI loading, channel management, orientation standardization, resampling, intensity scaling, and foreground cropping.
-   **Advanced Data Augmentation**: Incorporates `RandCropByPosNegLabeld` for balanced patch sampling and `RandAffined` for geometric augmentation to improve model generalization.
-   **Training and Validation Loop**: Provides a complete training script with Dice Loss and Dice Metric for performance monitoring.
-   **Reproducibility**: Uses a fixed random seed and clear logging for consistent results.
-   **TensorBoard Integration**: Logs training progress and metrics for visualization with TensorBoard.
-   **Modular Code Structure**: Organized into `src/dataset.py`, `src/model.py`, `src/train.py`, and `src/utils.py` for clarity and maintainability.

## Installation

To set up the project environment, follow these steps:

1.  **Clone the repository**:

    ```bash
    git clone https://github.com/your_username/monai_spleen_segmentation.git
    cd monai_spleen_segmentation
    ```

2.  **Create a Python virtual environment** (recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

    *Note: This project was developed with `monai==1.1.0` and `pytorch>=1.9`. Ensure your environment meets these requirements.*

## Usage

### 1. Download and Prepare Dataset

The `main.py` script will automatically download and extract the Medical Segmentation Decathlon Task 09 (Spleen) dataset into the `./data` directory if it's not already present. This dataset is approximately 300MB.

### 2. Train the Model

To start the training process, run the `main.py` script. You can specify various parameters:

```bash
python main.py \
    --data_dir ./data \
    --model_dir ./models \
    --log_dir ./runs \
    --epochs 600 \
    --batch_size 1 \
    --learning_rate 1e-4 \
    --seed 0
```

-   `--data_dir`: Path where the dataset will be stored/downloaded. (Default: `./data`)
-   `--model_dir`: Directory to save the best performing model checkpoint. (Default: `./models`)
-   `--log_dir`: Directory for TensorBoard logs. (Default: `./runs`)
-   `--epochs`: Number of training epochs. (Default: `600`)
-   `--batch_size`: Batch size for training and validation. (Default: `1`)
-   `--learning_rate`: Learning rate for the Adam optimizer. (Default: `1e-4`)
-   `--seed`: Random seed for reproducibility. (Default: `0`)
-   `--no_cuda`: Use this flag to force CPU training even if CUDA is available.

Training progress, including loss and Dice metric, will be logged to the console and TensorBoard.

### 3. Monitor Training with TensorBoard

While training is in progress, you can monitor its progress by launching TensorBoard:

```bash
tensorboard --logdir=./runs
```

Then, open your web browser and navigate to the address provided by TensorBoard (usually `http://localhost:6006`).

### 4. Inference (Future Work / Extension)

This project currently focuses on the training pipeline. For inference on new data, you would typically:

1.  Load the trained `best_metric_model.pth`.
2.  Apply the same preprocessing transforms as used during validation.
3.  Use `sliding_window_inference` for robust inference on large 3D volumes.
4.  Apply post-processing (e.g., argmax for segmentation maps).

## Project Structure

```
monai_spleen_segmentation/
├── data/                 # Dataset will be downloaded here
├── models/               # Trained model checkpoints will be saved here
├── runs/                 # TensorBoard logs will be stored here
├── src/
│   ├── __init__.py       # Makes 'src' a Python package
│   ├── dataset.py        # Defines data loading and MONAI transforms
│   ├── model.py          # Defines the 3D UNet model architecture
│   ├── train.py          # Contains the training and validation loop
│   └── utils.py          # Utility functions (e.g., dataset download)
├── main.py               # Main script to run the training pipeline
├── requirements.txt      # Python dependencies
└── README.md             # Project overview and documentation
```

## Dataset Details

The Medical Segmentation Decathlon (MSD) Task 09 (Spleen) dataset consists of 61 abdominal CT scans, with corresponding spleen segmentation masks. The dataset is split into 41 training and 20 testing cases. The images are 3D NIfTI files, and the segmentation task is binary (spleen vs. background). More details can be found on the [Medical Segmentation Decathlon website](http://medicaldecathlon.com/) [1].

## Model Architecture

The project employs a 3D U-Net, a convolutional neural network architecture widely used for biomedical image segmentation. The U-Net is characterized by its U-shaped architecture, which allows it to capture both fine-grained details and global contextual information through its contracting and expansive paths. The specific implementation uses MONAI's `UNet` class with `spatial_dims=3` and `Norm.BATCH` for batch normalization.

## Training Details

-   **Loss Function**: Dice Loss, which is effective for segmentation tasks, especially with imbalanced classes.
-   **Optimizer**: Adam optimizer with a learning rate of `1e-4`.
-   **Metrics**: Dice Metric (excluding background) is used to evaluate segmentation performance during validation.
-   **Inference Strategy**: `sliding_window_inference` is used during validation to handle large 3D volumes that might not fit into GPU memory entirely, processing them in smaller overlapping windows.

## Contributing

Feel free to fork this repository, open issues, or submit pull requests to improve the project.

## License

This project is open-sourced under the MIT License. See the `LICENSE` file for more details.

## References

[1] Medical Segmentation Decathlon. Available at: [http://medicaldecathlon.com/](http://medicaldecathlon.com/)
