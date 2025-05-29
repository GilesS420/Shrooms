from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Laad het model en de kolomnamen
model = joblib.load("random_forest_model.pkl")
columns = joblib.load("model_columns.pkl")

app = FastAPI(
    title="Mushroom Classification API",
    description="""
    This API uses a Random Forest model trained on the UCI Mushroom dataset to classify mushrooms as edible or poisonous.

    ### Model Info:
    - **Algorithm**: RandomForestClassifier (GridSearchCV-tuned)
    - **Hyperparameters**: n_estimators, max_depth, min_samples_leaf
    - **Evaluation**: 5-fold cross-validation (accuracy), test accuracy also reported
    - **Input**: 22 categorical features (one-hot encoded internally)
    - **Output**: Class label — 'e' (edible) or 'p' (poisonous)
    """,
    version="1.0.0",
)

class MushroomFeatures(BaseModel):
    features: dict

    class Config:
        schema_extra = {
            "example": {
                "features": {
                    "cap-shape": "x",
                    "cap-surface": "s",
                    "cap-color": "n",
                    "bruises": "t",
                    "odor": "p",
                    "gill-attachment": "f",
                    "gill-spacing": "c",
                    "gill-size": "n",
                    "gill-color": "k",
                    "stalk-shape": "e",
                    "stalk-root": "e",
                    "stalk-surface-above-ring": "s",
                    "stalk-surface-below-ring": "s",
                    "stalk-color-above-ring": "w",
                    "stalk-color-below-ring": "w",
                    "veil-type": "p",
                    "veil-color": "w",
                    "ring-number": "o",
                    "ring-type": "p",
                    "spore-print-color": "k",
                    "population": "s",
                    "habitat": "u"
                }
            }
        }

@app.post("/predict")
def predict_class(data: MushroomFeatures):
    # Zet de invoer om naar een DataFrame en pas one-hot encoding toe
    input_df = pd.DataFrame([data.features])
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=columns, fill_value=0)

    prediction = model.predict(input_df)
    return {"prediction": prediction[0]}

@app.get("/")
def root():
    return {"message": "Mushroom classification API - use POST /predict"}
