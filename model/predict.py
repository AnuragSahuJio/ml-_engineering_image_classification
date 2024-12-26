import joblib
from PIL import Image
import numpy as np

class ModelPredictor:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)

    def predict(self, img_path):
        img = Image.open(img_path).convert("RGB").resize((64, 64))
        img_array = np.array(img).flatten()
        prediction = self.model.predict([img_array])[0]
        probability = self.model.predict_proba([img_array])[0]
        return {
            "prediction": int(prediction),
            "probabilities": probability.tolist()
        }

if __name__ == "__main__":
    model = ModelPredictor('/workspaces/image_classification/model/image_svm.pkl')
    result = model.predict('/workspaces/image_classification/test/0/0.png')
    print(result)