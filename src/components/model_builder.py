from tensorflow.keras import Sequential, layers
from tensorflow.keras.applications import MobileNetV2

class ModelBuilder:
    @staticmethod
    def build_model(input_shape=(224, 224, 3), num_classes=2):
        # Load pre-trained MobileNetV2 as base model
        base_model = MobileNetV2(
            weights="imagenet",
            include_top=False,
            input_shape=input_shape
        )
        # Freeze base model layers
        base_model.trainable = False

        # Build full model
        model = Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax")
        ])
        return model

