import os
import shutil
import requests
import subprocess
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
FACE_DETECTOR_DIR = os.path.join(BASE_DIR, "face_detector")
TEMP_DIR = os.path.join(BASE_DIR, "temp_repo")

# URLs
REPO_URL = "https://github.com/prajnasb/observations.git"
PROTOTXT_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
MODEL_URL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

def create_dirs():
    os.makedirs(DATASET_DIR, exist_ok=True)
    os.makedirs(FACE_DETECTOR_DIR, exist_ok=True)

def download_file(url, output_path):
    if os.path.exists(output_path):
        logging.info(f"File already exists: {output_path}")
        return
    
    logging.info(f"Downloading {url} to {output_path}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logging.info("Download complete.")
    except Exception as e:
        logging.error(f"Failed to download {url}: {e}")

def setup_dataset():
    if os.path.exists(os.path.join(DATASET_DIR, "with_mask")) and \
       os.path.exists(os.path.join(DATASET_DIR, "without_mask")):
        logging.info("Dataset directories already exist. Skipping download.")
        return

    logging.info("Cloning dataset repository...")
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    
    try:
        logging.info(f"Cloning {REPO_URL} into {TEMP_DIR}...")
        subprocess.check_call(["git", "clone", "--depth", "1", REPO_URL, TEMP_DIR])
        
        # Paths in the repo - adjusting based on common structure in this specific repo
        # The repo structure is typically experiments/data/with_mask etc or similar
        # Let's search for them
        
        # Direct path to full dataset (identified via inspection)
        repo_mask_path = os.path.join(TEMP_DIR, "experiements/data/with_mask")
        repo_no_mask_path = os.path.join(TEMP_DIR, "experiements/data/without_mask")
             
        # Verify existence
        if not os.path.exists(repo_mask_path):
             # Try another common variant for this specific repo
             repo_mask_path = os.path.join(TEMP_DIR, "experiements/data/with_mask") # Typo in repo?
        
        # If still not found, just use the 'dest_mapping' one mentioned in my plan thoughts?
        # Let's stick to the prompt's implied structure or simple traversal.
        
        # Hardcoded fallback to the most likely location in 'prajnasb/observations'
        # It is usually: experiments/data/with_mask
        
        if os.path.exists(repo_mask_path) and os.path.exists(repo_no_mask_path):
            logging.info(f"Found data at {repo_mask_path} and {repo_no_mask_path}")
            shutil.copytree(repo_mask_path, os.path.join(DATASET_DIR, "with_mask"), dirs_exist_ok=True)
            shutil.copytree(repo_no_mask_path, os.path.join(DATASET_DIR, "without_mask"), dirs_exist_ok=True)
            logging.info("Dataset extracted successfully.")
        else:
            logging.error("Could not locate 'with_mask' and 'without_mask' directories in the cloned repo.")
            logging.info("Listing repo structure for debugging:")
            for root, dirs, files in os.walk(TEMP_DIR):
                logging.info(root)
            
    except Exception as e:
        logging.error(f"Error setting up dataset: {e}")
    finally:
        if os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR)
            logging.info("Cleaned up temp directory.")

def main():
    create_dirs()
    
    # Download Face Detector Models
    download_file(PROTOTXT_URL, os.path.join(FACE_DETECTOR_DIR, "deploy.prototxt"))
    download_file(MODEL_URL, os.path.join(FACE_DETECTOR_DIR, "res10_300x300_ssd_iter_140000.caffemodel"))
    
    # Setup Dataset
    setup_dataset()

if __name__ == "__main__":
    main()
