import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fastapi import APIRouter, UploadFile, HTTPException
from model.predict import ModelPredictor
from model.monitor import monitor_prediction_time

router = APIRouter()
predictor = ModelPredictor("model/image_svm.pkl")

@router.post("/predict/")
@monitor_prediction_time
async def predict(file: UploadFile):
    try:
        img_path = f"temp_{file.filename}"
        with open(img_path, "wb") as f:
            f.write(await file.read())
        result = predictor.predict(img_path)
        os.remove(img_path)  # Cleanup
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
