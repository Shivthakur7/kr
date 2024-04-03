from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Apple model
apple_model = tf.keras.models.load_model("./Apple.h5")
apple_model.compile(
    optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

# Load Potato model
potato_model = tf.keras.models.load_model("./Potato1.h5")
potato_model.compile(
    optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

# Load Tomato model
tomato_model = tf.keras.models.load_model("./Tomato1.h5")
tomato_model.compile(
    optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

# Load Corn model
corn_model = tf.keras.models.load_model("./Corn1.h5")
corn_model.compile(
    optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

# Load Grapes model
grapes_model = tf.keras.models.load_model("./Grapes.h5")
grapes_model.compile(
    optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

# Class names for each model
APPLE_CLASS_NAMES = ["Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy"]
POTATO_CLASS_NAMES = ["Potato___Early_blight", "Potato___healthy","Potato___Late_blight"]
TOMATO_CLASS_NAMES = ["Tomato___Bacterial_spot","Tomato___Early_blight","Tomato___Late_blight", "Tomato___Leaf_Mold","Tomato___Septoria_leaf_spot","Tomato___Spider_mites Two-spotted_spider_mite","Tomato___Target_Spot","Tomato___Tomato_Yellow_Leaf_Curl_Virus","Tomato___Tomato_mosaic_virus","Tomato___healthy"]
CORN_CLASS_NAMES = ["Corn_(maize)___Common_rust_","Corn_(maize)___Northern_Leaf_Blight","Corn_(maize)___healthy"]
GRAPES_CLASS_NAMES = ["Grape___Black_rot","Grape___Esca_(Black_Measles)","Grape___Leaf_blight_(Isariopsis_Leaf_Spot)","Grape___healthy"]

# Confidence threshold for predictions
CONFIDENCE_THRESHOLD = 0.81
# Expected input shape for the models
EXPECTED_SHAPE = (256, 256)

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data))
    # Resize the image to the expected input shape
    image = image.resize(EXPECTED_SHAPE)
    image = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
    return image

@app.post("/predict/apple")
async def predict_apple(
    file: UploadFile = File(..., media_type="image/jpeg")
):
    return await predict_disease(file, apple_model, APPLE_CLASS_NAMES)

@app.post("/predict/potato")
async def predict_potato(
    file: UploadFile = File(..., media_type="image/jpeg")
):
    return await predict_disease(file, potato_model, POTATO_CLASS_NAMES)

@app.post("/predict/tomato")
async def predict_tomato(
    file: UploadFile = File(..., media_type="image/jpeg")
):
    return await predict_disease(file, tomato_model, TOMATO_CLASS_NAMES)

@app.post("/predict/corn")
async def predict_corn(
    file: UploadFile = File(..., media_type="image/jpeg")
):
    return await predict_disease(file, corn_model, CORN_CLASS_NAMES)

@app.post("/predict/grapes")
async def predict_grapes(
    file: UploadFile = File(..., media_type="image/jpeg")
):
    return await predict_disease(file, grapes_model, GRAPES_CLASS_NAMES)

async def predict_disease(file, model, class_names):
    try:
        image = read_file_as_image(await file.read())
        img_batch = np.expand_dims(image, 0)
        predictions = model.predict(img_batch)

        confidence = np.max(predictions[0])

        # Check if the confidence is below the threshold
        if confidence < CONFIDENCE_THRESHOLD:
            # Handle uncertain predictions, for example, return a special label
            predicted_class = "Uncertain Prediction"
            response_data = {'class': predicted_class}
        else:
            predicted_class = class_names[np.argmax(predictions[0])]
            response_data = {'class': predicted_class, 'confidence': float(confidence)}

        return response_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=9000)
