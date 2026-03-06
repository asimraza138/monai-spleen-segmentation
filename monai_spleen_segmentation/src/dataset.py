
import os
import glob

from monai.transforms import (
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    RandCropByPosNegLabeld,
    RandAffined,
    Compose,
)
from monai.data import CacheDataset, DataLoader

def get_train_transforms(roi_size=(96, 96, 96), pixdim=(1.5, 1.5, 2.0)):
    """
    Defines the training transformations for the Spleen segmentation task.

    Args:
        roi_size (tuple): The spatial size of the random image patches to crop.
        pixdim (tuple): The pixel spacing for resampling.

    Returns:
        Compose: A composed MONAI transform object.
    """
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=pixdim,
                mode=("bilinear", "nearest"),
            ),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-57,
                a_max=164,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=roi_size,
                pos=1,
                neg=1,
                num_samples=4,
                random_center=True,
                random_size=False,
            ),
            RandAffined(
                keys=["image", "label"],
                mode=("bilinear", "nearest"),
                prob=0.1,
                spatial_size=roi_size,
                rotate_range=(0, 0, 0, 0, 0, 15),
                scale_range=(0.1, 0.1, 0.1),
                padding_mode="border",
            ),
        ]
    )

def get_val_transforms(roi_size=(96, 96, 96), pixdim=(1.5, 1.5, 2.0)):
    """
    Defines the validation transformations for the Spleen segmentation task.

    Args:
        roi_size (tuple): The spatial size of the random image patches to crop.
        pixdim (tuple): The pixel spacing for resampling.

    Returns:
        Compose: A composed MONAI transform object.
    """
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=pixdim,
                mode=("bilinear", "nearest"),
            ),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-57,
                a_max=164,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
        ]
    )

def get_data_loaders(data_dir, batch_size=1, roi_size=(96, 96, 96), pixdim=(1.5, 1.5, 2.0), cache_rate=0.1, num_workers=4):
    """
    Prepares and returns training and validation data loaders.

    Args:
        data_dir (str): Directory containing the dataset.
        batch_size (int): Batch size for the data loaders.
        roi_size (tuple): The spatial size of the random image patches to crop.
        pixdim (tuple): The pixel spacing for resampling.
        cache_rate (float): Percentage of cached data in CacheDataset.
        num_workers (int): Number of worker processes for data loading.

    Returns:
        tuple: A tuple containing (train_loader, val_loader).
    """
    images = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
    labels = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))
    data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(images, labels)]

    train_files, val_files = data_dicts[:-9], data_dicts[-9:]

    train_transforms = get_train_transforms(roi_size, pixdim)
    val_transforms = get_val_transforms(roi_size, pixdim)

    train_ds = CacheDataset(
        data=train_files,
        transform=train_transforms,
        cache_rate=cache_rate,
        num_workers=num_workers,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_ds = CacheDataset(
        data=val_files,
        transform=val_transforms,
        cache_rate=cache_rate,
        num_workers=num_workers,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers)

    return train_loader, val_loader

if __name__ == "__main__":
    # Example usage (for testing purposes)
    # This block will only run when dataset.py is executed directly
    print("Testing dataset.py components...")
    # Create dummy data_dir and dummy files for testing if needed
    # For a real test, you would need to download the MSD Spleen dataset
    # from http://medicaldecathlon.com/
    # For now, we'll just demonstrate the function calls.
    try:
        # Assuming a dummy 'data' directory exists with 'imagesTr' and 'labelsTr' subdirectories
        # and some .nii.gz files inside for a quick test.
        # In a real scenario, you'd replace 'dummy_data_path' with the actual path to your dataset.
        dummy_data_path = "./data/Task09_Spleen"
        if not os.path.exists(os.path.join(dummy_data_path, "imagesTr")):
            print(f"Warning: Dummy data path {dummy_data_path} not found. Cannot fully test data loaders.")
            print("Please download the Medical Segmentation Decathlon Spleen dataset and place it in the 'data' directory.")
        else:
            train_loader, val_loader = get_data_loaders(dummy_data_path)
            print(f"Successfully created train_loader with {len(train_loader.dataset)} samples.")
            print(f"Successfully created val_loader with {len(val_loader.dataset)} samples.")
            # You can add a loop here to iterate through a batch and print shapes
            # for batch_data in train_loader:
            #     print(f"Image batch shape: {batch_data['image'].shape}")
            #     print(f"Label batch shape: {batch_data['label'].shape}")
            #     break
    except Exception as e:
        print(f"An error occurred during testing: {e}")
    print("dataset.py testing complete.")
