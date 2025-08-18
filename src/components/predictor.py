import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from src.utils.common import load_model
from src.utils.common import mtcnn_detector



class Predictor:
    def __init__(self, model_path="models/mask_detector_best.h5"):
        self.model = load_model(model_path)
        self.mtcnn = mtcnn_detector
        self.class_labels = ["Mask", "No Mask"] 

    def predict_mask(self, face):
        # Load & preprocess image
        img = cv2.resize(face, (224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Predict
        preds = self.model.predict(img_array)
        class_idx = np.argmax(preds)
        class_label = self.class_labels[class_idx]
        score = preds[0][class_idx]

        # Filter only "Mask" predictions
        if class_label == "Mask" and score < 0.7:
            class_label = "Unknown"
            score = 1 - score  

        return class_label, score
        
    def face_detect(self, img_path, path=True, show=True, resize_factor=None): 
        """Detect faces, predict mask/no-mask, and annotate image."""
        if path: 
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                print(f"[ERROR] Could not read image from {img_path}")
                return
        else:
            img_bgr = img_path

         # Resize frame if requested (used for webcam optimization)
        if resize_factor is not None:
            small_frame = cv2.resize(img_bgr, (0, 0), fx=resize_factor, fy=resize_factor)
            img_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)    

        # Detect faces
        boxes, _ = self.mtcnn.detect(img_rgb)

        h, w, _ = img_bgr.shape
        margin_ratio = 0.25

        if boxes is not None:
            # Scale boxes back to original size if resized
            if resize_factor is not None:
                boxes = boxes / resize_factor
                
            for box in boxes:
                x1, y1, x2, y2 = box.astype(int)

                # Face dimensions
                face_w = x2 - x1
                face_h = y2 - y1

                # Apply margin
                margin_x = int(face_w * margin_ratio)
                margin_y = int(face_h * margin_ratio)

                x1_pad = max(0, x1 - margin_x)
                y1_pad = max(0, y1 - margin_y)
                x2_pad = min(w, x2 + margin_x)
                y2_pad = min(h, y2 + margin_y)

                # Crop face with margin
                face = img_bgr[y1_pad:y2_pad, x1_pad:x2_pad]
                    
                label, prob = self.predict_mask(face)

                # Draw bounding box + label
                if label == "Mask":
                    color = (0, 255, 0)       # Green
                elif label == "No Mask":
                    color = (0, 0, 255)       # Red
                else:  # Unknown
                    color = (0, 255, 255)                
                
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    img_bgr,
                    f"{label}: {prob:.2f}",
                    (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

        if show:
            cv2.imshow("Face Mask Detection", img_bgr)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            # For webcam, just return the annotated frame
            return img_bgr

    def face_detect_webcam(self,cam_index): 
        cap = cv2.VideoCapture(cam_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        print("[INFO] Webcam started. Press 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            annotated_frame = self.face_detect(frame, path=False, show=False,resize_factor=0.5)
            if annotated_frame is not None:
                cv2.imshow("Face Mask Detection", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] 'q' pressed. Exiting webcam.")
                break
        

        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Webcam closed successfully.")
