import pandas as pd
from joblib import load
from flask import Flask, request, jsonify
from flask_cors import CORS
from category_encoders import BinaryEncoder

model= load("decision_tree_model.joblib")

x=pd.read_csv("Employee_Attrition-Dataset.csv")

categorical_features = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'Over18', 'OverTime']
encoder = BinaryEncoder()
x_encoded = encoder.fit_transform(x[categorical_features])

api = Flask(__name__)
CORS(api)

@api.route('/api/employee_attrition', methods=['POST'])
def predict_employee_attrition():
    data = request.json['inputs']
    input_df = pd.DataFrame(data)
    
    # Encode categorical features
    input_encoded = encoder.transform(input_df[categorical_features])
    input_df = input_df.drop(categorical_features, axis=1)
    input_encoded = input_encoded.reset_index(drop=True)

    final_input = pd.concat([input_df, input_encoded], axis=1)
    
    prediction = model.predict_proba(final_input)
    class_labels = model.classes_

    response = []
    for prob in prediction:
        prob_dict = {}
        for k, v in zip(class_labels, prob):
            prob_dict[str(k)] = round(float(v) * 100, 2)
        response.append(prob_dict)
    
    # Return the prediction as a JSON response
    return jsonify({'prediction': response})

if __name__ == '__main__':
    api.run(port=8000)