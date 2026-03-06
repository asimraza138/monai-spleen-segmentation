
import argparse
import logging
import os
import torch

from src.train import train_model
from src.utils import download_and_extract_msd_spleen

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    parser = argparse.ArgumentParser(description="MONAI Spleen Segmentation Project")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Directory to store the dataset. Will download if not present.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./models",
        help="Directory to save trained models.",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./runs",
        help="Directory to save TensorBoard logs.",
    )
    parser.add_argument(
        "--epochs", type=int, default=600, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for training and validation."
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate for the optimizer."
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Disable CUDA even if available."
    )

    args = parser.parse_args()

    # Determine device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    logging.info(f"Using device: {device}")

    # Download and extract dataset
    try:
        logging.info("Checking for dataset...")
        msd_spleen_dir = download_and_extract_msd_spleen(root_dir=args.data_dir)
        logging.info(f"Dataset ready at: {msd_spleen_dir}")
    except Exception as e:
        logging.error(f"Failed to download or extract dataset: {e}")
        return

    # Start training
    logging.info("Starting model training...")
    best_metric, best_metric_epoch = train_model(
        data_dir=msd_spleen_dir,
        model_dir=args.model_dir,
        log_dir=args.log_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=device,
        seed=args.seed,
    )
    logging.info(
        f"Training complete. Best validation Dice metric: {best_metric:.4f} at epoch {best_metric_epoch}"
    )


if __name__ == "__main__":
    main()
