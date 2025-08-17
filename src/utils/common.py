import os
import numpy as np
import tensorflow as tf
import torch
from facenet_pytorch import MTCNN


def save_model(model, model_path="models/mask_detector_best.h5"):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)

def load_model(model_path="models/mask_detector_best.h5"):
    return tf.keras.models.load_model(model_path)

# Choose device for MTCNN
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[INFO] MTCNN running on device: {device}")

# Initialize MTCNN once with GPU if available
mtcnn_detector = MTCNN(keep_all=True, device=device)
