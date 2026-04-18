import os
import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image

# Configuration
IMG_SIZE = (224, 224)
# Use relative paths for portability
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'database', 'Braintumor', 'Training')
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'brain_tumor.tflite')
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']

def predict_image(image_path):
    """
    Predicts the class of a given image using TFLite model.
    """
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}")
        return {"error": "Model file missing on server."}

    try:
        # Load TFLite model and allocate tensors
        interpreter = tflite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()

        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Preprocess image using PIL instead of keras
        img = Image.open(image_path).convert('RGB')
        img = img.resize(IMG_SIZE)
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array.astype(np.float32)
        
        # Standardize based on EfficientNetV2 requirements (usually [0, 255] for some versions, or [0, 1])
        # We use [0, 1] as it was used during conversion
        img_array /= 255.0

        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], img_array)

        # Run inference
        interpreter.invoke()

        # Get output tensor
        prediction = interpreter.get_tensor(output_details[0]['index'])
        
        predicted_class_index = np.argmax(prediction)
        confidence = float(np.max(prediction))
        predicted_class = CLASSES[predicted_class_index]
        
        return {
            "class": predicted_class,
            "confidence": confidence,
            "probabilities": {cls: float(prob) for cls, prob in zip(CLASSES, prediction[0])}
        }
    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}

if __name__ == "__main__":
    print("TFLite Inference Module for Brain Tumor Detection")
