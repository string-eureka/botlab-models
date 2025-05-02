from flask import Blueprint, request, jsonify
from services.clip_service import call_clip_predict

clip_bp = Blueprint("clip", __name__)

@clip_bp.route("/predict", methods=["POST"])
def clip_predict():
    data = request.get_json()
    return jsonify(call_clip_predict(data))
