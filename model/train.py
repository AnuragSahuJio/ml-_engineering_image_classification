import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
from PIL import Image

def load_data(data_dir):
    """
    Load images and labels from the directory.

    Args:
        data_dir (str): Directory containing class-wise images.

    Returns:
        tuple: (X, y) where X is the image data and y are the labels.
    """
    X, y = [], []
    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            label = int(class_name.split("_")[-1])
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                img = Image.open(img_path).convert("RGB").resize((64, 64))
                X.append(np.array(img).flatten())  # Flatten image
                y.append(label)
    return np.array(X), np.array(y)

def train_model(data_dir, model_path):
    X, y = load_data(data_dir)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train SVM
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)
    
    # Test the model
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    
    # Save the model
    joblib.dump(model, model_path)
    print(f"Model saved at {model_path}")

if __name__ == "__main__":
    train_model("data/images", "model/image_svm.pkl")
