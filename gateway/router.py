from flask import Blueprint, request, jsonify
from services import clip_service, sam_service

router = Blueprint("router", __name__)

@router.route("/clip/predict", methods=["POST"])
def clip_predict():
    data = request.get_json()
    image = data.get("image")
    classes = data.get("classes")
    if not image or not isinstance(classes, list):
        return jsonify({"error": "Missing or invalid 'image' or 'classes'"}), 400
    return jsonify(clip_service.predict_clip(image, classes))


@router.route("/sam/predict", methods=["POST"])
def sam_predict():
    data = request.get_json()
    image = data.get("image")
    if not image:
        return jsonify({"error": "Missing 'image'"}), 400
    return jsonify(sam_service.predict_masks(image))


@router.route("/sam/predict_with_points", methods=["POST"])
def sam_predict_points():
    data = request.get_json()
    image = data.get("image")
    points = data.get("points")
    labels = data.get("labels")
    if not image or not points or not labels:
        return jsonify({"error": "Missing 'image', 'points', or 'labels'"}), 400
    return jsonify(sam_service.predict_with_points(image, points, labels))


@router.route("/sam/predict_with_bbox", methods=["POST"])
def sam_predict_bbox():
    data = request.get_json()
    image = data.get("image")
    bbox = data.get("bbox")
    if not image or not bbox:
        return jsonify({"error": "Missing 'image' or 'bbox'"}), 400
    return jsonify(sam_service.predict_with_bbox(image, bbox))
