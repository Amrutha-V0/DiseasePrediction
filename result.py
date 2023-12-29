import streamlit as st

def main(predictions):
    st.title('Prediction Result')
    st.write('Predicted Disease')
    st.write(f"RF Model Prediction: {predictions['rf_model_prediction']}")
    st.write(f"Naive Bayes Prediction: {predictions['naive_bayes_prediction']}")
    st.write(f"SVM Model Prediction: {predictions['svm_model_prediction']}")
    st.write(f"Final Prediction: {predictions['final_prediction']}")

if __name__ == '__main__':
    # Example predictions (replace with actual predictions)
    example_predictions = {
        "rf_model_prediction": "ExamplePrediction",
        "naive_bayes_prediction": "ExamplePrediction",
        "svm_model_prediction": "ExamplePrediction",
        "final_prediction": "FinalExamplePrediction"
    }
    main(example_predictions)
