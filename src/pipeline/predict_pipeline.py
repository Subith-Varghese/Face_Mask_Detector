from src.components.predictor import Predictor

if __name__ == "__main__":
    # Initialize predictor with trained model
    predictor = Predictor("models/mask_detector_best.h5")

    while True:
        print("\nPress 'w' for Webcam, 'p' for Picture, or 's' to Skip...")
        choice = input("Your choice: ").strip().lower()

        if choice == "w":
            predictor.face_detect_webcam()
            break

        elif choice == "p":
            img_path = input("Enter image path: ").strip()
            predictor.face_detect(img_path)
            break

        elif choice == "s":
            print("Exiting program.")
            break

        else:
            print("❌ Invalid choice! Please try again.")
