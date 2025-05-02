import base64
import io
import torch
import uuid


from helper import download_image,delete_images
from flask import Flask, request, Response, jsonify
from PIL import Image
from urllib.parse import urlparse
from transformers import CLIPProcessor, CLIPModel

app = Flask(__name__)

# Comment out to enable debug mode
# logging.basicConfig(level=logging.DEBUG)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


@app.route("/health", methods=["GET"])
def ping():
    return Response(response="Clip is up!", status=200, mimetype="application/json")


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

    for field in ["image","classes"]:
        if field not in data:
            return Response(
                response='{"reason": "Request does not contain {field}"}',
                status=400,
                mimetype="application/json",
            )
     
    try:
        im = data["image"]
        if urlparse(im).scheme in ("http", "https", "file"):
            image = download_image(request_id, im)
        else:
            image = Image.open(io.BytesIO(base64.b64decode(im))).convert("RGB")

    except Exception as e:
        return Response(
            response = f"Unable to download image, error {e}",
            status = 500,
            mimetype = "application/json"
        )
        
    try:
        text_prompts = [f"a photo of a {label}" for label in data["classes"]]
        inputs = processor(text=text_prompts, images=image, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=-1)

        values, indices = probs[0].topk(1)

        top_indices = indices.tolist()
        top_classes = [data["classes"][i] for i in top_indices]
        top_scores = values.tolist()

        return jsonify({
            "indices": top_indices,
            "classes": top_classes,
            "scores": top_scores
        })
        
    except Exception as e:
        return Response(
            response=f'Unable to process request, error : {e} "',
            status=500,
            mimetype="application/json",
        )
        
    finally:
        delete_images(request_id)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
    
    
"""

curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{
        "image": "https://example.com/sample_image.jpg",
        "classes": ["dog", "cat", "car", "tree"]

      }'
"""