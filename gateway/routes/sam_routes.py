from flask import Blueprint, request, jsonify
from services.sam_service import (
    call_sam_predict,
    call_sam_predict_with_points,
    call_sam_predict_with_bbox
)

sam_bp = Blueprint("sam", __name__)

@sam_bp.route("/predict", methods=["POST"])
def sam_predict():
    data = request.get_json()
    return jsonify(call_sam_predict(data))

@sam_bp.route("/predict_with_points", methods=["POST"])
def sam_predict_with_points():
    data = request.get_json()
    return jsonify(call_sam_predict_with_points(data))

@sam_bp.route("/predict_with_bbox", methods=["POST"])
def sam_predict_with_bbox():
    data = request.get_json()
    return jsonify(call_sam_predict_with_bbox(data))
