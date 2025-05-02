import os
import requests

CLIP_URL = os.getenv("CLIP_SERVICE_URL")

def call_clip_predict(data):
    try:
        response = requests.post(CLIP_URL, json=data)
        return response.json()
    except Exception as e:
        return {"error": str(e)}
