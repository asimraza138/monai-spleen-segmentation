
import os
import torch
import logging
from datetime import datetime

from monai.data import decollate_batch
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism

from torch.utils.tensorboard import SummaryWriter

from src.dataset import get_data_loaders
from src.model import get_unet_model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_model(
    data_dir,
    model_dir="./models",
    log_dir="./runs",
    epochs=600,
    batch_size=1,
    learning_rate=1e-4,
    roi_size=(96, 96, 96),
    pixdim=(1.5, 1.5, 2.0),
    cache_rate=0.1,
    num_workers=4,
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    seed=0,
):
    """
    Trains a 3D UNet model for spleen segmentation.

    Args:
        data_dir (str): Path to the dataset directory (e.g., containing Task09_Spleen).
        model_dir (str): Directory to save trained models.
        log_dir (str): Directory to save TensorBoard logs.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training and validation.
        learning_rate (float): Learning rate for the optimizer.
        roi_size (tuple): Region of interest size for cropping.
        pixdim (tuple): Pixel dimensions for resampling.
        cache_rate (float): Cache rate for MONAI CacheDataset.
        num_workers (int): Number of worker processes for data loading.
        device (torch.device): Device to run training on (cuda or cpu).
        seed (int): Random seed for reproducibility.
    """

    logging.info(f"Starting training on device: {device}")
    set_determinism(seed=seed)

    # Create directories if they don't exist
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Setup TensorBoard writer
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=os.path.join(log_dir, f"spleen_segmentation_{current_time}"))

    # Get data loaders
    train_loader, val_loader = get_data_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        roi_size=roi_size,
        pixdim=pixdim,
        cache_rate=cache_rate,
        num_workers=num_workers,
    )
    logging.info(f"Data loaders initialized. Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")

    # Initialize model, loss, optimizer, and metric
    model = get_unet_model(in_channels=1, out_channels=2).to(device)
    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    best_metric = -1
    best_metric_epoch = -1

    for epoch in range(epochs):
        logging.info(f"Epoch {epoch + 1}/{epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            logging.debug(f"Train Step {step}/{len(train_loader)} Loss: {loss.item():.4f}")
        epoch_loss /= step
        writer.add_scalar("training_loss", epoch_loss, epoch)
        logging.info(f"Epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % 10 == 0:  # Evaluate every 10 epochs
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_labels = val_data["image"].to(device), val_data["label"].to(device)
                    roi_size_inference = (160, 160, 160) # Larger ROI for inference
                    val_outputs = sliding_window_inference(
                        val_inputs, roi_size_inference, sw_batch_size=4, predictor=model
                    )
                    val_outputs = [decollate_batch(val_outputs)]
                    val_labels = [decollate_batch(val_labels)]
                    dice_metric(y_pred=val_outputs, y=val_labels)

                metric = dice_metric.aggregate().item()
                dice_metric.reset()
                writer.add_scalar("val_dice_metric", metric, epoch)
                logging.info(f"Epoch {epoch + 1} average Dice metric: {metric:.4f}")

                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(model_dir, "best_metric_model.pth"))
                    logging.info("Saved new best metric model")

    logging.info(
        f"Training completed. Best metric: {best_metric:.4f} at epoch {best_metric_epoch}"
    )
    writer.close()
    return best_metric, best_metric_epoch

if __name__ == "__main__":
    # This block will only run when train.py is executed directly
    print("Testing train.py components...")
    # For a real test, you would need to download the MSD Spleen dataset
    # and place it in the specified data_dir.
    # Example: python train.py --data_dir /path/to/your/Task09_Spleen
    import argparse

    parser = argparse.ArgumentParser(description="Train MONAI Spleen Segmentation Model")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the Medical Segmentation Decathlon Spleen dataset")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs for testing")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--model_dir", type=str, default="./models", help="Directory to save models")
    parser.add_argument("--log_dir", type=str, default="./runs", help="Directory to save TensorBoard logs")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")

    args = parser.parse_args()

    try:
        print(f"Attempting to train with data from: {args.data_dir}")
        best_metric, best_epoch = train_model(
            data_dir=args.data_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            model_dir=args.model_dir,
            log_dir=args.log_dir,
            seed=args.seed,
            # Use CPU for testing if CUDA is not available or for quick checks
            device=torch.device("cpu") # Force CPU for quick local testing without GPU
        )
        print(f"Test training finished. Best Dice Metric: {best_metric:.4f} at Epoch: {best_epoch}")
    except Exception as e:
        print(f"An error occurred during training test: {e}")
        print("Please ensure the dataset is correctly downloaded and placed in the specified data_dir.")
        

