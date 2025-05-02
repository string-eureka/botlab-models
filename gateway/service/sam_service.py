import os
import requests

SAM_BASE = os.getenv("SAM_BASE_URL")

def call_sam_predict(data):
    return _post(f"{SAM_BASE}/predict", data)

def call_sam_predict_with_points(data):
    return _post(f"{SAM_BASE}/predict_with_points", data)

def call_sam_predict_with_bbox(data):
    return _post(f"{SAM_BASE}/predict_with_bbox", data)

def _post(url, data):
    try:
        response = requests.post(url, json=data)
        return response.json()
    except Exception as e:
        return {"error": str(e)}
