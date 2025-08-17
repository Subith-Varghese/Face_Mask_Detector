import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

class DataIngestion:
    def __init__(self, base_dir, img_size=(224, 224), batch_size=32):
        self.base_dir = base_dir
        self.train_dir = os.path.join(base_dir, "Train")
        self.val_dir = os.path.join(base_dir, "Validation")
        self.test_dir = os.path.join(base_dir, "Test")
        self.img_size = img_size
        self.batch_size = batch_size

    def get_data_generators(self):
        # Data augmentation for training
        train_aug = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=20,
            zoom_range=0.15,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.15,
            horizontal_flip=True
        )

        # For validation/test → only preprocessing
        val_aug = ImageDataGenerator(preprocessing_function=preprocess_input)

        # Training generator
        train_gen = train_aug.flow_from_directory(
            self.train_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode="categorical"
        )

        # Validation generator
        val_gen = val_aug.flow_from_directory(
            self.val_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode="categorical"
        )

        # Test generator
        test_gen = val_aug.flow_from_directory(
            self.test_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode="categorical",
            shuffle=False
        )

        return train_gen, val_gen, test_gen
