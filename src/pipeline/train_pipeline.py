from src.components.data_downloader import DataDownloader
from src.components.data_ingestion import DataIngestion
from src.components.model_builder import ModelBuilder
from src.components.model_trainer import ModelTrainer
from src.utils.logger import logger


if __name__ == "__main__":
    try:
        # Step 1: Download dataset
        dataset_url = "https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset"
        downloader = DataDownloader(dataset_url, "data/")
        logger.info(f"📥 Downloading dataset from {dataset_url} ...")
        downloader.download()
        logger.info(f"✅ Dataset downloaded successfully.")

        # Step 2: Dataset base directory
        base_dir = "data/face-mask-12k-images-dataset/Face Mask Dataset"
        logger.info(f"Dataset base directory: {base_dir}")

        # Step 3: Data ingestion (train/val/test)
        data_ingestion = DataIngestion(base_dir, img_size=(224, 224), batch_size=32)
        train_gen, val_gen, test_gen = data_ingestion.get_data_generators()
        logger.info("✅ Data generators created successfully.")

        # Step 4: Build & Train model
        model = ModelBuilder.build_model(input_shape=(224, 224, 3), num_classes=2)
        logger.info("✅ Model built successfully.")

        # Step 5: Train model
        trainer = ModelTrainer(model, train_gen, val_gen, epochs=20)
        logger.info("🚀 Starting model training...")
        trainer.train()
        logger.info("✅ Training complete. Model saved successfully.")
        
    except Exception as e:
        logger.error(f"❌ Training pipeline failed: {e}")
        print(f"[ERROR] Training pipeline failed: {e}")