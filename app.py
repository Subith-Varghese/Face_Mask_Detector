from flask import Flask, render_template, request, redirect
from src.components.predictor import Predictor
from src.utils.logger import logger

app = Flask(__name__)

# Initialize predictor
predictor = Predictor("models/mask_detector_best.h5")

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/upload", methods=["GET", "POST"])
def upload_image():
    from flask import request, url_for
    import os, cv2
    UPLOAD_FOLDER = "static/uploads"
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    if request.method == "POST":
        try:
            file = request.files.get("image")
            if not file or file.filename == "":
                return render_template("upload.html", error="No file selected")
            
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            annotated_img = predictor.face_detect(filepath, path=True, show=False)
            if annotated_img is None:
                return render_template("upload.html", error="No human face detected! Please try another image.")
            
            output_path = os.path.join(UPLOAD_FOLDER, "annotated_" + file.filename)
            cv2.imwrite(output_path, annotated_img)
            return render_template("upload.html", uploaded_image=output_path)

        except Exception as e:
            logger.error(f"Upload failed: {e}")
            return render_template("upload.html", error=str(e))

    return render_template("upload.html")

@app.route("/webcam")
def webcam_page():
    return render_template("webcam.html")

@app.route("/start_webcam", methods=["POST"])
def start_webcam():
    try:
        cam_index = int(request.form.get("cam_index", 0))
        logger.info(f"Starting webcam detection on index {cam_index}")
        predictor.face_detect_webcam(cam_index)
        return "Webcam detection finished. Check your OpenCV window."
    except Exception as e:
        logger.error(f"Webcam detection failed: {e}")
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)
