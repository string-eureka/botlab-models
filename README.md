## Setup Instructions

### 1. Prerequisites

- Docker and Docker Compose installed

### 2. Build & Run

```bash
docker compose up --build
```

### 3. Running Services

| Service  | URL                     | Description          |
|----------|-------------------------|----------------------|
| Gateway  | http://localhost:5000   | Unified API endpoint |
| CLIP     | http://localhost:5001   | Visual classifier    |
| SAM      | http://localhost:5002   | Image segmentation   |

---

## 🔌 Unified API Endpoints (via Gateway)

All client requests go through: `http://localhost:5000/`

### CLIP Model

#### `POST /clip/predict`

Classifies an image based on candidate labels.

**Request**
```json
{
  "image": "https://example.com/sample_image.jpg",
  "classes": ["dog", "cat", "car", "tree"]
}
```

**Response**
```json

{
    "classes":["dog"],
    "indices":[0],
    "scores":[0.9989804625511169]
}
```

---

### SAM (Segment Anything Model)

#### `POST /sam/predict`  
Returns all segmentation masks for an image.

**Request**
```json
{
  "image": "https://example.com/sample_image.jpg"
}
```

---

#### `POST /sam/predict_with_points`  
Returns masks guided by point prompts.

**Request**
```json
{
  "image": "https://example.com/sample_image.jpg",
  "points": [[500, 375], [250, 200]],
  "labels": [1, 0]
}
```

---

#### `POST /sam/predict_with_bbox`  
Returns masks guided by bounding box.

**Request**
```json
{
  "image": "https://example.com/sample_image.jpg",
  "bbox": [100, 100, 400, 400]
}
```

---

## Adding a New API or Model

### A. Add a New API to an Existing Model

1. **Edit the Model’s Code**
   - Add a new route to `clip/app.py` or `sam/app.py`.

2. **Expose It Through the Gateway**
   - Update or create a service method in `gateway/services/<model>_service.py`.
   - Add a route in `gateway/model_router.py` to call the service.

3. **Rebuild**
   ```bash
   docker-compose up --build
   ```

---

### B. Add a New Model (e.g., DepthAnything)

1. **Create a New Model Folder**
   ```bash
   mkdir depth
   cp -r clip/* depth/
   ```

2. **Implement Your Model in `depth/app.py`**

3. **Create a Service File**
   Create `gateway/services/depth_service.py` with logic to call the new API.

4. **Add Routes**
   Update `gateway/model_router.py` with new `/depth/...` routes.

5. **Update Docker Compose**
   ```yaml
   services:
     depth:
       build: ./depth
       ports:
         - "5003:5003"
   ```

6. **Rebuild and Start**
   ```bash
   docker-compose up --build
   ```

---


## TODO

1. Add support for the depthanything model
2. Add CLI support to create APIs for client provided models
