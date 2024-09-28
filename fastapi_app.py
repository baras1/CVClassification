from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import numpy as np
import torch
from PIL import Image
import io
from classifier_manager import ClassifierManager

app = FastAPI()

# Load the models once at startup
cnn_manager = ClassifierManager(model_type='cnn', num_classes=75)
cnn_manager.load_model('models/cnn_model.pth')

# mlp_manager = ClassifierManager(model_type='mlp', input_size=128*128*3, hidden_size=128, output_size=75)
# mlp_manager.load_model('models/mlp_model')

# cnn_numpy_manager = ClassifierManager(model_type='cnn_numpy', input_shape=(3, 128, 128), num_classes=75)
# cnn_numpy_manager.load_model('models/cnn_numpy_model')

@app.post("/predict/")
async def predict(image: UploadFile = File(...), model_type: str = Form(...)):
    contents = await image.read()
    img = Image.open(io.BytesIO(contents)).convert('RGB')
    img = img.resize((128, 128))  # Resize to the required input shape
    img_np = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
    img_np = img_np.transpose((2, 0, 1))  # Change shape to (C, H, W)
    img_np = img_np[None, :]  # Add batch dimension

    if model_type == 'cnn':
        result = cnn_manager.predict(torch.tensor(img_np).float())  # Convert to tensor for PyTorch
    # elif model_type == 'mlp':
    #     img_np_flat = img_np.flatten().reshape(1, -1)  # Flatten for MLP
    #     result = mlp_manager.predict(img_np_flat)
    # elif model_type == 'cnn_numpy':
    #     result = cnn_numpy_manager.predict(img_np)

    return JSONResponse({"prediction": int(result[0])})
