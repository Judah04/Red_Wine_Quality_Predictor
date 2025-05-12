import gradio as gr
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle

# Load saved model and scaler
model = pickle.load(open("red_wine_model.pkl", "rb"))
scaler = pickle.load(open("red_wine_scaler.pkl", "rb"))


# Gradio prediction function
def predict_wine_quality(
    fixed_acidity,
    volatile_acidity,
    citric_acid,
    residual_sugar,
    chlorides,
    free_sulfur_dioxide,
    total_sulfur_dioxide,
    density,
    pH,
    sulphates,
    alcohol,
):

    input_data = [
        [
            fixed_acidity,
            volatile_acidity,
            citric_acid,
            residual_sugar,
            chlorides,
            free_sulfur_dioxide,
            total_sulfur_dioxide,
            density,
            pH,
            sulphates,
            alcohol,
        ]
    ]

    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)
    return f"Predicted Wine Quality: {prediction[0]}"


# Create Gradio interface
interface = gr.Interface(
    fn=predict_wine_quality,
    inputs=[
        gr.Number(label="Fixed Acidity"),
        gr.Number(label="Volatile Acidity"),
        gr.Number(label="Citric Acid"),
        gr.Number(label="Residual Sugar"),
        gr.Number(label="Chlorides"),
        gr.Number(label="Free Sulfur Dioxide"),
        gr.Number(label="Total Sulfur Dioxide"),
        gr.Number(label="Density"),
        gr.Number(label="pH"),
        gr.Number(label="Sulphates"),
        gr.Number(label="Alcohol"),
    ],
    outputs="text",
    title="Red Wine Quality Predictor",
    description="Enter wine chemical properties to predict its quality.",
)

interface.launch()
