# 🩺 Predictive Pulse - Blood Pressure Prediction

Predictive Pulse is a Flask web application designed to predict the blood pressure stage of users based on their symptoms, health history, and lifestyle inputs. This tool aims to provide users with insights into their blood pressure status, promoting better health management and awareness.

---

## 👥 Team

- **Atul Sharma** 
- Devang Gupta
- Farhan Akhtar

**College**: Anand International College of Engineering, Jaipur

---

## 🧠 Tech Used

- **Programming Language**: Python
- **Web Framework**: Flask
- **Data Manipulation**: Pandas
- **Machine Learning**: Scikit-learn
- **Model Serialization**: Pickle (for saving trained models)

---

## 📁 Project Files

- `app.py` – The main Flask application file that handles user requests and serves the web interface.
- `train_model.py` – Script for training the machine learning model using the provided dataset.
- `model_artifacts/` – Directory containing saved models and encoders for preprocessing input data.
- `templates/index.html` – HTML file for the web user interface, where users can input their data.
- `static/style.css` – Stylesheet for customizing the appearance of the web application.
- `patient_data.csv` – Dataset used for training the model, containing various patient attributes and their corresponding blood pressure stages.
- `requirements.txt` – A file listing all the necessary Python packages required to run the application.

---

## 🚀 How to Run the App

To run the Predictive Pulse application locally, follow these steps:

1.  **Clone the Repository**:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Install Required Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Flask Application**:
    ```bash
    python app.py
    ```

4.  Access the Application: [Click here](https://5799dff6-3309-4739-a27a-86a281a72773-00-ewvip8m3q4gv.sisko.replit.dev) to open the deployed app in your browser.

