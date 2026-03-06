
import os
import tarfile
import hashlib
import requests
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_and_extract_msd_spleen(root_dir="./data", resource="https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar", md5="410d4a301d4e3b2f6f66ec3ddba524e"):
    """
    Downloads and extracts the Medical Segmentation Decathlon (MSD) Spleen dataset.

    Args:
        root_dir (str): The directory where the dataset will be stored.
        resource (str): The URL to the dataset tar file.
        md5 (str): The MD5 checksum of the tar file for integrity verification.

    Returns:
        str: The path to the extracted dataset directory.
    """
    dataset_dir = os.path.join(root_dir, "Task09_Spleen")
    if os.path.exists(dataset_dir):
        logging.info(f"Dataset already exists at {dataset_dir}. Skipping download and extraction.")
        return dataset_dir

    os.makedirs(root_dir, exist_ok=True)
    compressed_file_path = os.path.join(root_dir, "Task09_Spleen.tar")

    logging.info(f"Downloading dataset from {resource} to {compressed_file_path}")
    try:
        response = requests.get(resource, stream=True)
        response.raise_for_status()  # Raise an exception for HTTP errors
        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024 # 1 Kibibyte
        t = tqdm(total=total_size, unit="iB", unit_scale=True)
        with open(compressed_file_path, "wb") as f:
            for data in response.iter_content(block_size):
                t.update(len(data))
                f.write(data)
        t.close()
        logging.info("Download complete.")
    except requests.exceptions.RequestException as e:
        logging.error(f"Error downloading file: {e}")
        if os.path.exists(compressed_file_path):
            os.remove(compressed_file_path)
        raise

    # Verify MD5 checksum
    logging.info("Verifying MD5 checksum...")
    with open(compressed_file_path, "rb") as f:
        current_md5 = hashlib.md5(f.read()).hexdigest()
    if current_md5 != md5:
        logging.error(f"MD5 checksum mismatch. Expected {md5}, got {current_md5}.")
        os.remove(compressed_file_path)
        raise ValueError("MD5 checksum mismatch.")
    logging.info("MD5 checksum verified.")

    logging.info(f"Extracting {compressed_file_path} to {root_dir}")
    try:
        with tarfile.open(compressed_file_path, "r") as tar:
            tar.extractall(path=root_dir)
        logging.info("Extraction complete.")
    except tarfile.ReadError as e:
        logging.error(f"Error extracting tar file: {e}")
        os.remove(compressed_file_path)
        raise
    finally:
        # Clean up the compressed file after extraction
        if os.path.exists(compressed_file_path):
            os.remove(compressed_file_path)

    return dataset_dir


if __name__ == "__main__":
    print("Testing utils.py components...")
    try:
        # This will attempt to download and extract the dataset
        # Be aware that this downloads a large file (~300MB)
        downloaded_path = download_and_extract_msd_spleen()
        print(f"Dataset available at: {downloaded_path}")
        # You can add checks here to verify the presence of expected files/directories
        if os.path.exists(os.path.join(downloaded_path, "imagesTr")) and \
           os.path.exists(os.path.join(downloaded_path, "labelsTr")):
            print("Dataset structure appears correct.")
        else:
            print("Warning: Dataset structure might be incomplete or incorrect.")
    except Exception as e:
        print(f"An error occurred during utils testing: {e}")
    print("utils.py testing complete.")
