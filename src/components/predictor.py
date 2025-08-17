import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from src.utils.common import load_model
from src.utils.common import mtcnn_detector
from collections import Counter
from sort_tracker.sort import Sort




class Predictor:
    def __init__(self, model_path="models/mask_detector_best.h5"):
        self.model = load_model(model_path)
        self.mtcnn = mtcnn_detector
        self.tracker = Sort()
        # Hard-coded mapping (consistent with dataset)
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
        return class_label,preds[0][class_idx]
    
    def face_detect(self,img_path): 
        """Detect faces, predict mask/no-mask, and annotate image."""
        
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
       
        # Detect faces
        boxes, _ = self.mtcnn.detect(img_rgb)

            # Image dimensions
        h, w, _ = img_bgr.shape
        margin_ratio = 0.15

        if boxes is not None:
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
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 1)
                cv2.putText(
                    img_bgr,
                    f"{label}: {prob:.2f}",
                    (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1,
                )

        cv2.imshow("Face Mask Detection", img_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        

    
    def face_detect_webcam(self): 
        # Initialize webcam
        cap = cv2.VideoCapture(1)   # 0 = default webcam
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        print("[INFO] Webcam started. Press 'q' to quit.")

        # For smoothing predictions
        recent_predictions_dict = {}
        N = 5

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Annotate frame
            annotated_frame = frame.copy()
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect faces
            boxes, probs  = self.mtcnn.detect(img_rgb)
            detections = []
            if boxes is not None:
                for box, prob in zip(boxes, probs):
                    x1, y1, x2, y2 = box
                    detections.append([x1, y1, x2, y2, prob])

            detections = np.array(detections)

            if detections.size == 0:
                detections = np.empty((0, 5))

            # Update SORT tracker with detections
            tracked_objects = self.tracker.update(detections)

            h, w, _ = frame.shape
            margin_ratio = 0.25  # smaller margin for webcam

            for x1, y1, x2, y2,score, obj_id in tracked_objects:
                face_w, face_h = x2 - x1, y2 - y1

                # Apply margin
                margin_x = int(face_w * margin_ratio)
                margin_y = int(face_h * margin_ratio)
                x1_pad = max(0, x1 - margin_x)
                y1_pad = max(0, y1 - margin_y)
                x2_pad = min(w, x2 + margin_x)
                y2_pad = min(h, y2 + margin_y)

                # Crop face & convert to RGB
                face = frame[int(y1_pad):int(y2_pad), int(x1_pad):int(x2_pad)]
                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

                # Predict mask
                label, prob = self.predict_mask(face_rgb)

                # Initialize list for this face if not exist
                if obj_id not in recent_predictions_dict:
                    recent_predictions_dict[obj_id] = []

                # Append label for this face
                recent_predictions_dict[obj_id].append(label)
                if len(recent_predictions_dict[obj_id]) > N:
                    recent_predictions_dict[obj_id].pop(0)

                label = Counter(recent_predictions_dict[obj_id]).most_common(1)[0][0]
                # Draw bounding box + label
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                cv2.rectangle(annotated_frame, (int(x1), int(y1)),(int(x2), int(y2)), color, 2)
                cv2.putText(
                    annotated_frame,
                    f"{int(obj_id)}:{label}: {prob:.2f}",
                    (int(x1), int(y1) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

            # Show annotated frame
            cv2.imshow("Face Mask Detection", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] 'q' pressed. Exiting webcam.")
                break

        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Webcam closed successfully.")
