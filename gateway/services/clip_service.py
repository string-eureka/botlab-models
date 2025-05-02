from utils.requester import post_json

CLIP_BASE_URL = "http://clip:5001"

def predict_clip(image_url, classes):
    return post_json(f"{CLIP_BASE_URL}/predict", {
        "image": image_url,
        "classes": classes
    })
