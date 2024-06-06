import os
import pandas as pd
import joblib
from pandas import Index
from PyQt5.QtWidgets import QWidget, QPushButton, QLabel, QLineEdit, QPushButton, QFormLayout, QMessageBox

class PredictorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.load_model()
        self.initialize_gui()

    def load_model(self):
        try:
            dirname = os.path.dirname(__file__)
            model_path = os.path.join(dirname, '../models/diabetes_model_nb.pkl')
            self.model = joblib.load(model_path)
            print("Models loaded successfully.")

            self.feature_columns = [
                'Pregnancies', 
                'Glucose', 
                'BloodPressure', 
                'SkinThickness', 
                'Insulin', 
                'BMI',
                'DiabetesPedigreeFunction', 
                'Age']
            
            self.target_column = 'Outcome'

        except FileNotFoundError as e:
            print(f"Error: {e}")
            raise

    def initialize_gui(self):
        self.setWindowTitle('Diabetes Prediction')
        self.create_form()

    def create_form(self):
        layout = QFormLayout()
        
        self.pregnancyInputLabel = QLabel('Pregnancies (0 - 17):')
        self.pregnancyInput = QLineEdit()
        layout.addRow(self.pregnancyInputLabel, self.pregnancyInput)

        self.glucoseInputLabel = QLabel('Glucose (0 - 200):')
        self.glucoseInput = QLineEdit()
        layout.addRow(self.glucoseInputLabel, self.glucoseInput)

        self.bloodPressureInputLabel = QLabel('Blood Pressure (0 - 140):')
        self.bloodPressureInput = QLineEdit()
        layout.addRow(self.bloodPressureInputLabel, self.bloodPressureInput)

        self.skinThicknessInputLabel = QLabel('Skin Thickness (0 - 99):')
        self.skinThicknessInput = QLineEdit()
        layout.addRow(self.skinThicknessInputLabel, self.skinThicknessInput)

        self.insulinInputLabel = QLabel('Insulin (0 - 2000):')
        self.insulinInput = QLineEdit()
        layout.addRow(self.insulinInputLabel, self.insulinInput)

        self.bmiInputLabel = QLabel('BMI (0 - 70):')
        self.bmiInput = QLineEdit()
        layout.addRow(self.bmiInputLabel, self.bmiInput)

        self.dpfInputLabel = QLabel('Diabetes Pedigree Function (0 - 3):')
        self.dpfInput = QLineEdit()
        layout.addRow(self.dpfInputLabel, self.dpfInput)

        self.ageInputLabel = QLabel('Age (15 - 100):')
        self.ageInput = QLineEdit()
        layout.addRow(self.ageInputLabel, self.ageInput)

        self.submitButton = QPushButton('Submit')
        self.submitButton.clicked.connect(self.submit_form)
        layout.addRow(self.submitButton)

        self.setLayout(layout)

    def submit_form(self):
        result = self.predict_outcome()
        QMessageBox.information(self, 'Prediction', f'{result}')
        
        # Clear the form fields
        # self.pregnancyInput.clear()

    def get_form_data(self):
        form_data = [
            float(self.get_input_value(self.pregnancyInput.text())),
            float(self.get_input_value(self.glucoseInput.text())),
            float(self.get_input_value(self.bloodPressureInput.text())),
            float(self.get_input_value(self.skinThicknessInput.text())),
            float(self.get_input_value(self.insulinInput.text())),
            float(self.get_input_value(self.bmiInput.text())),
            float(self.get_input_value(self.dpfInput.text())),
            float(self.get_input_value(self.ageInput.text())),
        ]

        return pd.DataFrame([form_data], columns=Index(self.feature_columns))

    def get_input_value(self, value):
        return 0.0 if value is None or not value.isdigit() else value

    def predict_outcome(self):
        prediction = int(self.model.predict(self.get_form_data())[0])
        print(self.get_form_data())
        result = "Diabetic" if prediction == 1 else "Non-Diabetic"
        return result