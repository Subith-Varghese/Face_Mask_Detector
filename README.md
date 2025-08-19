#  Face Mask Detection (Deep Learning + Flask + OpenCV)

A complete end-to-end project for **real-time face mask detection** using **Deep Learning (MobileNetV2)**, **MTCNN (Face Detection)**, and a **Flask Web App**.  
The system can detect whether a person is wearing a **mask** or **no mask** from images or webcam in real time.  

---

## 🚀 Features
- 📥 **Automatic dataset download** from Kaggle.  
- 🖼️ **Image upload** for mask detection.  
- 🎥 **Real-time webcam detection** with OpenCV.  
- 🧑‍💻 **Deep learning model** using MobileNetV2 (transfer learning).  
- 📊 **Data augmentation & preprocessing** with TensorFlow/Keras.  
- 📝 **Logging support** for debugging and monitoring.  
- 🌐 **Flask web interface** with HTML templates.  

---

## 📂 Project Structure
  
  face_mask_detector/
  │── data/ # Dataset directory
  │── models/ # Saved trained models
  │── notebooks/ # Jupyter notebooks for experiments
  │── src/
  │ ├── components/ # Core ML components
  │ │ ├── data_downloader.py
  │ │ ├── data_ingestion.py
  │ │ ├── model_builder.py
  │ │ ├── model_trainer.py
  │ │ └── predictor.py
  │ │
  │ ├── pipeline/ # Training & prediction pipelines
  │ │ ├── train_pipeline.py
  │ │ └── predict_pipeline.py
  │ │
  │ └── utils/ # Utility scripts
  │ ├── common.py
  │ └── logger.py
  │
  │── templates/ # Flask HTML templates
  │ ├── home.html
  │ ├── upload.html
  │ └── webcam.html
  │
  │── app.py # Flask web app
  │── requirements.txt # Python dependencies
  │── README.md # Project documentation
