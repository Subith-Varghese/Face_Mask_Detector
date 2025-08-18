import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from src.utils.common import save_model
from src.utils.logger import logger
import os


class ModelTrainer:
    def __init__(self, model, train_gen, val_gen, epochs=20, init_lr=1e-4, model_dir="models/"):
        self.model = model
        self.train_gen = train_gen
        self.val_gen = val_gen
        self.epochs = epochs
        self.init_lr = init_lr
        self.model_dir = model_dir

        os.makedirs(self.model_dir, exist_ok=True)
        self.model_path = os.path.join(self.model_dir, "mask_detector_best.h5")

    def train(self):
        try:
            # Compile with Adam optimizer & custom learning rate
            self.model.compile(
                optimizer=Adam(learning_rate=self.init_lr),
                loss="categorical_crossentropy",
                metrics=["accuracy"]
            )
            logger.info("🚀 Starting training...")
            # Define callbacks
            checkpoint = ModelCheckpoint(
                self.model_path,
                monitor="val_accuracy",
                save_best_only=True,
                verbose=1
            )
            early_stop = EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
                verbose=1
            )

            history = self.model.fit(
                self.train_gen,
                validation_data=self.val_gen,
                epochs=self.epochs,
                callbacks=[checkpoint, early_stop]
            )

            logger.info(f"✅ Training complete. Best model saved at {self.model_path}")
            return history
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise e