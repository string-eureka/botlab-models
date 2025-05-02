import os
from urllib.parse import urlparse
import requests
from PIL import Image
import logging
import shutil
IMAGES_FOLDER = "./temp"

def download_image(request_id, image_url):
    folder = os.path.join(IMAGES_FOLDER, request_id)
    os.makedirs(folder, exist_ok=True)

    filename = os.path.join(folder, os.path.basename(urlparse(image_url).path))
    try:
        response = requests.get(image_url)
        with open(filename, "wb") as f:
            f.write(response.content)
        return Image.open(filename).convert("RGB")
    except Exception as e:
        raise RuntimeError(f"Failed to download image from {image_url}: {e}")

def delete_images(request_id):
    directory = os.path.join(IMAGES_FOLDER, request_id)
    try:
        shutil.rmtree(directory)
    except OSError as e:
        logging.error(f"Error deleting image directory {directory}: {e}")