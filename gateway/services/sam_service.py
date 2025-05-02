from utils.requester import post_json

SAM_BASE_URL = "http://sam:5002"

def predict_masks(image_url):
    return post_json(f"{SAM_BASE_URL}/predict", {"image": image_url})

def predict_with_points(image_url, points, labels):
    return post_json(f"{SAM_BASE_URL}/predict_with_points", {
        "image": image_url,
        "points": points,
        "labels": labels
    })

def predict_with_bbox(image_url, bbox):
    return post_json(f"{SAM_BASE_URL}/predict_with_bbox", {
        "image": image_url,
        "bbox": bbox
    })
