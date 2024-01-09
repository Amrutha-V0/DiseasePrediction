from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load models and necessary components from the pickle file
with open('models.pkl', 'rb') as f:
    saved_models = pickle.load(f)
    final_rf_model = saved_models['rf_model']
    final_nb_model = saved_models['nb_model']
    final_svm_model = saved_models['svm_model']
    data_dict = saved_models['data_dict']
    encoder = saved_models['encoder']

# Home route - displays the form to input symptoms
@app.route('/')
def front_page():
    return render_template('front_page.html')

# Route to display the form for symptom input
@app.route('/home.html')
def home():
    return render_template('home.html')

# Predict route - receives symptoms and predicts disease
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        symptoms = request.form['symptoms']

        # Process symptoms and make predictions
        input_data = [0] * len(data_dict["symptom_index"])
        for symptom in symptoms.split(","):
            if symptom.capitalize() in data_dict["symptom_index"]:
                index = data_dict["symptom_index"][symptom.capitalize()]
                input_data[index] = 1

        input_data = [input_data]  # Reshape data for prediction
        rf_prediction = encoder.inverse_transform(final_rf_model.predict(input_data))[0]
        nb_prediction = encoder.inverse_transform(final_nb_model.predict(input_data))[0]
        svm_prediction = encoder.inverse_transform(final_svm_model.predict(input_data))[0]

        predictions = {
            "rf_model_prediction": rf_prediction,
            "naive_bayes_prediction": nb_prediction,
            "svm_model_prediction": svm_prediction,
            "final_prediction": max(set([rf_prediction, nb_prediction, svm_prediction]), key=[rf_prediction, nb_prediction, svm_prediction].count)
        }
        return render_template('result.html', predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)
