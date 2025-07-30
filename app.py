from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)

# Load model and encoders
model_path = os.path.join("model_artifacts", "model.pkl")
encoders_path = os.path.join("model_artifacts", "encoders.pkl")
target_encoder_path = os.path.join("model_artifacts", "target_encoder.pkl")

model = joblib.load(model_path)
encoders = joblib.load(encoders_path)
target_encoder = joblib.load(target_encoder_path)

@app.route('/', methods=['GET'])  # Allow only GET here
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])  # Form submits here using POST
def predict():
    data = {
        'C': request.form['Gender'],
        'Age': request.form['Age'],
        'History': request.form['History'],
        'Patient': request.form['Patient'],
        'TakeMedication': request.form['TakeMedication'],
        'Severity': request.form['Severity'],
        'BreathShortness': request.form['BreathShortness'],
        'VisualChanges': request.form['VisualChanges'],
        'NoseBleeding': request.form['NoseBleeding'],
        'Whendiagnoused': request.form['Whendiagnoused'],
        'Systolic': float(request.form['Systolic']),
        'Diastolic': float(request.form['Diastolic']),
        'ControlledDiet': request.form['ControlledDiet']
    }

    # Apply encoding
    for col in encoders:
        data[col] = encoders[col].transform([data[col]])[0]

    input_data = [[
        data['C'], data['Age'], data['History'], data['Patient'],
        data['TakeMedication'], data['Severity'], data['BreathShortness'],
        data['VisualChanges'], data['NoseBleeding'], data['Whendiagnoused'],
        data['Systolic'], data['Diastolic'], data['ControlledDiet']
    ]]

    prediction = model.predict(input_data)
    prediction_label = target_encoder.inverse_transform(prediction)[0]

    return render_template('index.html', prediction_text=f"Predicted BP Stage: {prediction_label}")

if __name__ == '__main__':
    app.run(debug=True)
