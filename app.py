from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io
import os

# Environment variables
MODEL_PATH = os.environ.get("MODEL_PATH", "PneumoScanModel/pneumonia_detection_ai_version_2.h5")
PORT = int(os.environ.get("PORT", 5000))

app = FastAPI()

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://pneumoscan-tl2m.onrender.com/"],  # frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load Keras model
model = load_model(MODEL_PATH)
print("Model loaded successfully!")
print("Model input shape:", model.input_shape)

# Prediction endpoint
@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert("L")  # grayscale
        img = img.resize((200, 200))  # adjust to your model input
        x = np.array(img, dtype=np.float32) / 255.0
        x = np.expand_dims(x, axis=-1)  # (H,W,1)
        x = np.expand_dims(x, axis=0)   # (1,H,W,1)

        preds = model.predict(x)
        pneumonia_prob = float(preds[0][0])
        normal_prob = 1 - pneumonia_prob
        result_label = "PNEUMONIA" if pneumonia_prob >= normal_prob else "NORMAL"

        return {
            "result": result_label,
            "probability":max(pneumonia_prob,normal_prob),
            "pneumonia_probability": pneumonia_prob,
            "normal_probability": normal_prob
        }

    except Exception as e:
        return {"error": str(e)}

# Serve frontend (React build)
frontend_path = os.environ.get("FRONTEND_PATH", "../PneumoScan/dist")
app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
