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
```  
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
```


---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/face_mask_detector.git
cd face_mask_detector
```
---
2️⃣ Create virtual environment

```
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows
```
---
3️⃣ Install dependencies
```
pip install -r requirements.txt
```
---
4️⃣ Run training pipeline (optional if model not trained)
```
python src/pipeline/train_pipeline.py
```
---
5️⃣ Run Flask app
```
python app.py
```
Visit 👉 http://127.0.0.1:5000/
 in your browser.
---
🖥️ Usage
- Home Page: Choose to upload an image or use webcam.
- Upload Image: Upload an image, model detects faces & masks, annotated image shown on screen.
- Webcam Detection: Select webcam index (0 = default, 1 = external) → start live detection.

---
🔄 Project Flow
``
flowchart TD
    A[Start] --> B[Download Dataset from Kaggle]
    B --> C[Data Ingestion: Train/Val/Test Generators]
    C --> D[Build CNN Model: MobileNetV2 + Custom Layers]
    D --> E[Train Model: Checkpoint + EarlyStopping]
    E --> F[Save Best Model]
    F --> G[Prediction Phase]
    G --> H{Choose Mode}
    H -->|Upload Image| I[Face Detection with MTCNN + Mask Prediction]
    I -->|Webcam| J[Real-time Face Detection + Mask Prediction]
    J --> K[Annotated Image Returned]
    K --> K
    L --> L[Flask Web App Displays Result]
``
