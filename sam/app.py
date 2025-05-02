import base64
import io
import torch
import uuid

import numpy as np

from helper import download_image, delete_images
from flask import Flask, request, Response, jsonify
from PIL import Image
from urllib.parse import urlparse

from segment_anything import SamPredictor,sam_model_registry

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_type = "vit_b"
checkpoint_path = "sam_vit_b_01ec64.pth" 
sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
sam.to(device=device)
predictor = SamPredictor(sam)


@app.route("/health", methods=["GET"])
def ping():
    return Response(response="SAM v2 is up!", status=200, mimetype="application/json")


def encode_binary_mask(mask):
    """Encodes a binary mask as base64 string for transmission"""
    mask_bytes = (mask * 255).astype(np.uint8).tobytes()
    return base64.b64encode(mask_bytes).decode('utf-8')


def encode_masks(masks: np.ndarray):
    """Encodes multiple masks with their scores and bounding boxes"""
    result = []
    
    if len(masks.shape) == 3:
        num_masks = masks.shape[0]
        
        for i in range(num_masks):
            mask = masks[i]
            if mask.sum() > 0:
                y_indices, x_indices = np.where(mask > 0)
                bbox = [int(y_indices.min()), int(x_indices.min()), 
                        int(y_indices.max()), int(x_indices.max())]
                
                result.append({
                    "mask": encode_binary_mask(mask),
                    "bbox": bbox,
                    "area": int(mask.sum())
                })
    
    return result


@app.route("/predict", methods=["POST"])
def predict():
    if request.content_type != "application/json":
        return Response(
            response='{"reason": "Content-type must be application/json"}',
            status=400,
            mimetype="application/json",
        )

    data = request.get_json()
    request_id = uuid.uuid4().hex

    if "image" not in data:
        return Response(
            response='{"reason": "Request does not contain image"}',
            status=400,
            mimetype="application/json",
        )
     
    try:
        im = data["image"]
        if urlparse(im).scheme in ("http", "https", "file"):
            image = download_image(request_id, im)
        else:
            image = Image.open(io.BytesIO(base64.b64decode(im))).convert("RGB")
            
        image_array = np.array(image)

    except Exception as e:
        return Response(
            response=f'{{"reason": "Unable to download or process image, error: {str(e)}"}}',
            status=500,
            mimetype="application/json",
        )
        
    try:
        predictor.set_image(image_array)
        masks, scores, _ = predictor.predict()
        mask_data = encode_masks(masks)
        
        return jsonify({
            "request_id": request_id,
            "num_masks": len(mask_data),
            "masks": mask_data,
            "scores": scores.tolist() if isinstance(scores, np.ndarray) else scores,
        })
        
    except Exception as e:
        return Response(
            response=f'{{"reason": "Unable to process request, error: {str(e)}"}}',
            status=500,
            mimetype="application/json",
        )
        
    finally:
        delete_images(request_id)


@app.route("/predict_with_points", methods=["POST"])
def predict_with_points():
    """Endpoint for predicting masks with point prompts"""
    if request.content_type != "application/json":
        return Response(
            response='{"reason": "Content-type must be application/json"}',
            status=400,
            mimetype="application/json",
        )

    data = request.get_json()
    request_id = uuid.uuid4().hex

    required_fields = ["image", "points", "labels"]
    for field in required_fields:
        if field not in data:
            return Response(
                response=f'{{"reason": "Request does not contain {field}"}}',
                status=400,
                mimetype="application/json",
            )
     
    try:
        im = data["image"]
        if urlparse(im).scheme in ("http", "https", "file"):
            image = download_image(request_id, im)
        else:
            image = Image.open(io.BytesIO(base64.b64decode(im))).convert("RGB")
            
        image_array = np.array(image)
        input_points = np.array(data["points"])
        input_labels = np.array(data["labels"])

    except Exception as e:
        return Response(
            response=f'{{"reason": "Unable to download or process image, error: {str(e)}"}}',
            status=500,
            mimetype="application/json",
        )
        
    try:
        predictor.set_image(image_array)
        
        masks, scores, _ = predictor.predict(
            point_coords=input_points,
            point_labels=input_labels
        )
        mask_data = encode_masks(masks)
        
        return jsonify({
            "request_id": request_id,
            "num_masks": len(mask_data),
            "masks": mask_data,
            "scores": scores.tolist() if isinstance(scores, np.ndarray) else scores,
        })
        
    except Exception as e:
        return Response(
            response=f'{{"reason": "Unable to process request, error: {str(e)}"}}',
            status=500,
            mimetype="application/json",
        )
        
    finally:
        delete_images(request_id)


@app.route("/predict_with_bbox", methods=["POST"])
def predict_with_bbox():
    """Endpoint for predicting masks with bounding box prompts"""
    if request.content_type != "application/json":
        return Response(
            response='{"reason": "Content-type must be application/json"}',
            status=400,
            mimetype="application/json",
        )

    data = request.get_json()
    request_id = uuid.uuid4().hex

    required_fields = ["image", "bbox"]
    for field in required_fields:
        if field not in data:
            return Response(
                response=f'{{"reason": "Request does not contain {field}"}}',
                status=400,
                mimetype="application/json",
            )
     
    try:
        im = data["image"]
        if urlparse(im).scheme in ("http", "https", "file"):
            image = download_image(request_id, im)
        else:
            image = Image.open(io.BytesIO(base64.b64decode(im))).convert("RGB")
            
        image_array = np.array(image)
        bbox = np.array(data["bbox"])

    except Exception as e:
        return Response(  
            response=f'{{"reason": "Unable to download or process image, error: {str(e)}"}}',
            status=500,
            mimetype="application/json",
        )
        
    try:
        predictor.set_image(image_array)
        
        masks, scores, _ = predictor.predict(
            box=bbox
        )        
        mask_data = encode_masks(masks)
        
        return jsonify({
            "request_id": request_id,
            "num_masks": len(mask_data),
            "masks": mask_data,
            "scores": scores.tolist() if isinstance(scores, np.ndarray) else scores,
        })
        
    except Exception as e:
        return Response(
            response=f'{{"reason": "Unable to process request, error: {str(e)}"}}',
            status=500,
            mimetype="application/json",
        )
        
    finally:
        delete_images(request_id)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)


"""
Examples

1. Get all masks for an image:
curl -X POST http://localhost:5002/predict \
  -H "Content-Type: application/json" \
  -d '{
        "image": "https://example.com/sample_image.jpg"
      }'

2. Using point prompts:
curl -X POST http://localhost:5002/predict_with_points \
  -H "Content-Type: application/json" \
  -d '{
        "image": "https://example.com/sample_image.jpg",
        "points": [[500, 375], [250, 200]],
        "labels": [1, 0]
      }'

3. Using bounding box:
curl -X POST http://localhost:5002/predict_with_bbox \
  -H "Content-Type: application/json" \
  -d '{
        "image": "https://example.com/sample_image.jpg",
        "bbox": [100, 100, 400, 400]
      }'
"""