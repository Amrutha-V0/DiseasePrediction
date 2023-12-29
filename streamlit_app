import streamlit as st
import pickle

# Load models and necessary components from the pickle file
@st.cache(allow_output_mutation=True)
def load_models():
    with open('models.pkl', 'rb') as f:
        saved_models = pickle.load(f)
        final_rf_model = saved_models['rf_model']
        final_nb_model = saved_models['nb_model']
        final_svm_model = saved_models['svm_model']
        data_dict = saved_models['data_dict']
        encoder = saved_models['encoder']
    return final_rf_model, final_nb_model, final_svm_model, data_dict, encoder

def main():
    st.title('Symptom Predictor')
    final_rf_model, final_nb_model, final_svm_model, data_dict, encoder = load_models()

    st.write('Enter Symptoms')
    symptoms = st.text_input('Symptoms (comma-separated)')

    if st.button('Predict'):
        input_data = [0] * len(data_dict["symptom_index"])
        for symptom in symptoms.split(","):
            if symptom.capitalize() in data_dict["symptom_index"]:
                index = data_dict["symptom_index"][symptom.capitalize()]
                input_data[index] = 1

        input_data = [input_data]  # Reshape data for prediction
        rf_prediction = encoder.inverse_transform(final_rf_model.predict(input_data))[0]
        nb_prediction = encoder.inverse_transform(final_nb_model.predict(input_data))[0]
        svm_prediction = encoder.inverse_transform(final_svm_model.predict(input_data))[0]

        final_prediction = max(set([rf_prediction, nb_prediction, svm_prediction]), key=[rf_prediction, nb_prediction, svm_prediction].count)

        st.write('Predicted Disease')
        st.write(f"RF Model Prediction: {rf_prediction}")
        st.write(f"Naive Bayes Prediction: {nb_prediction}")
        st.write(f"SVM Model Prediction: {svm_prediction}")
        st.write(f"Final Prediction: {final_prediction}")

if __name__ == '__main__':
    main()
