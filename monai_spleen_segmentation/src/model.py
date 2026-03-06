
from monai.networks.nets import UNet
from monai.networks.layers import Norm

def get_unet_model(in_channels=1, out_channels=2, channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2)):
    """
    Defines and returns a 3D UNet model for medical image segmentation.

    Args:
        in_channels (int): Number of input channels (e.g., 1 for grayscale CT/MRI).
        out_channels (int): Number of output channels (e.g., 2 for background and foreground).
        channels (tuple): Tuple of feature map sizes for each layer.
        strides (tuple): Tuple of stride values for downsampling.

    Returns:
        UNet: A MONAI UNet model instance.
    """
    model = UNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels,
        strides=strides,
        num_res_units=2,
        norm=Norm.BATCH,
    )
    return model

if __name__ == "__main__":
    # Example usage (for testing purposes)
    print("Testing model.py components...")
    try:
        model = get_unet_model()
        print(f"Successfully created UNet model with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters.")
        # You can add a dummy input tensor and pass it through the model to test its output shape
        # import torch
        # dummy_input = torch.randn(1, 1, 96, 96, 96) # Batch, Channel, Depth, Height, Width
        # output = model(dummy_input)
        # print(f"Output shape for dummy input: {output.shape}")
    except Exception as e:
        print(f"An error occurred during testing: {e}")
    print("model.py testing complete.")
