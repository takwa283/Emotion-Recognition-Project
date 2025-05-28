from fastapi import FastAPI, UploadFile
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()

# ✅ Charger le modèle TFLite
def load_model():
    model_path = "face.tflite"  # Assurez-vous que ce fichier est dans le même dossier
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        print("✅ Modèle chargé avec succès !")
        return interpreter
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle : {e}")
        return None

interpreter = load_model()

# ✅ Liste des émotions
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

@app.post("/predict")
async def predict_emotion(file: UploadFile):
    if not interpreter:
        return {"error": "❌ Aucun modèle chargé !"}

    try:
        # ✅ Charger et prétraiter l'image
        image_data = await file.read()
        img = Image.open(io.BytesIO(image_data)).convert("RGB")
        img = img.resize((224, 224))
        img_array = np.array(img, dtype=np.float32) / 255.0  # Normalisation

        # ✅ Ajouter une dimension pour correspondre à l'entrée du modèle
        img_array = np.expand_dims(img_array, axis=0)

        # ✅ Récupérer les détails du modèle
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # ✅ Faire la prédiction
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]

        # ✅ Trouver l'émotion avec la probabilité maximale
        best_emotion = EMOTIONS[np.argmax(predictions)]
        confidence = float(np.max(predictions)) * 100

        return {
            "emotion_predite": best_emotion,
            "confiance": confidence,
            "predictions": {EMOTIONS[i]: float(predictions[i]) * 100 for i in range(len(EMOTIONS))}
        }

    except Exception as e:
        return {"error": str(e)}

