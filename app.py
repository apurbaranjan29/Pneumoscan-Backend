from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io
import os

app = FastAPI()

# Allow CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Keras model
MODEL_PATH = "PneumoScanModel/pneumonia_detection_ai_version_2.h5"
model = load_model(MODEL_PATH)
print("Model loaded successfully!")
print("Model input shape:", model.input_shape)

# Prediction endpoint
@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    try:
        # Read and preprocess image
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert("L")  # grayscale
        img = img.resize((200, 200))  # adjust to your model input
        x = np.array(img, dtype=np.float32) / 255.0
        x = np.expand_dims(x, axis=-1)  # (H,W,1)
        x = np.expand_dims(x, axis=0)   # (1,H,W,1)
        
        preds = model.predict(x)
        # Run prediction
        pneumonia_prob = float(preds[0][0])
        print(pneumonia_prob)   # sigmoid output = PNEUMONIA probability
        normal_prob = 1 - pneumonia_prob  
        print(normal_prob)      # NORMAL probability
        max_probability=max(pneumonia_prob,normal_prob)
        print(max_probability)
        # Determine label
        if pneumonia_prob >= normal_prob:
            result_label = "PNEUMONIA"
        else:
            result_label = "NORMAL"

        print(result_label)

        return {
    "result": result_label,
    "probability": max(pneumonia_prob, normal_prob),
    "pneumonia_probability": pneumonia_prob,
    "normal_probability": normal_prob
}


    except Exception as e:
        return {"error": str(e)}

# Serve frontend (React build)
frontend_path = os.path.join(os.path.dirname(__file__), "../PneumoScan/dist")
app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
