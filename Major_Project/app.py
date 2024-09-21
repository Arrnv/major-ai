import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('my_model.h5')

# Function to preprocess input data
def preprocess_input(df):
    # Standardize numerical columns
    scaler = StandardScaler()
    numerical_cols = ['height', 'weight', 'ap_hi', 'ap_lo', 'age_years', 'bmi']
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    return df

# Function to predict cardiovascular disease
def predict_cardiovascular_disease(model, input_data):
    # Preprocess input data
    input_data_processed = preprocess_input(input_data)
    
    # Ensure input data has the correct shape and type
    features = input_data_processed.values.reshape(1, -1).astype(np.float32)
    
    # Predict
    prediction = model.predict(features)
    
    return prediction

# Main function to run the Streamlit app
def main():
    st.title('Cardiovascular Disease Prediction App')
    st.write('This app predicts the likelihood of having cardiovascular disease based on input data.')
    
    # Sidebar
    st.sidebar.header('User Input Features')
    
    # Collect user input features
    height = st.sidebar.slider('Height (cm)', 100, 250, 150)
    weight = st.sidebar.slider('Weight (kg)', 30.0, 200.0, 70.0)
    ap_hi = st.sidebar.slider('Systolic blood pressure (mmHg)', 60, 250, 120)
    ap_lo = st.sidebar.slider('Diastolic blood pressure (mmHg)', 40, 150, 80)
    age_years = st.sidebar.slider('Age (years)', 18, 100, 50)
    bmi = weight / ((height / 100) ** 2)
    gender_2 = st.sidebar.checkbox('Gender: Male')
    cholesterol_2 = st.sidebar.checkbox('Cholesterol Level: Above Normal')
    cholesterol_3 = st.sidebar.checkbox('Cholesterol Level: Well Above Normal')
    gluc_2 = st.sidebar.checkbox('Glucose Level: Above Normal')
    gluc_3 = st.sidebar.checkbox('Glucose Level: Well Above Normal')
    smoke_1 = st.sidebar.checkbox('Smoker')
    alco_1 = st.sidebar.checkbox('Alcoholic')
    active_1 = st.sidebar.checkbox('Active Lifestyle')
    bp_encoded_1_0 = st.sidebar.checkbox('Blood Pressure Category: Normal')
    bp_encoded_2_0 = st.sidebar.checkbox('Blood Pressure Category: Above Normal')
    bp_encoded_3_0 = st.sidebar.checkbox('Blood Pressure Category: High')
    
    # Create a DataFrame for the input data
    input_data = pd.DataFrame({
        'height': [height],
        'weight': [weight],
        'ap_hi': [ap_hi],
        'ap_lo': [ap_lo],

        'age_years': [age_years],
        'bmi': [bmi],
        'gender_2': [gender_2],
        'cholesterol_2': [cholesterol_2],
        'cholesterol_3': [cholesterol_3],
        'gluc_2': [gluc_2],
        'gluc_3': [gluc_3],
        'smoke_1': [smoke_1],
        'alco_1': [alco_1],
        'active_1': [active_1],
        'bp_encoded_1.0': [bp_encoded_1_0],
        'bp_encoded_2.0': [bp_encoded_2_0],
        'bp_encoded_3.0': [bp_encoded_3_0],
        'bmi_category_Obese': [bmi >= 30],
        'bmi_category_Overweight': [25 <= bmi < 30],
        'bmi_category_Underweight': [bmi < 18.5]
    })
    
    # Predict
    if st.sidebar.button('Predict'):
        prediction = predict_cardiovascular_disease(model, input_data)
        prediction = prediction*100
        prediction_text = "Yes" if prediction >= 0.5 else "No"
        st.write(f"Does person have risk of cardiac arrest  {prediction_text}")

# Run the app
if __name__ == '__main__':
    main()



# import tensorflow as tf

# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(64, activation='relu', input_dim=X_train.shape[1], kernel_regularizer=tf.keras.regularizers.l2(0.01)),
#     tf.keras.layers.Dropout(0.3),
#     tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
#     tf.keras.layers.Dropout(0.3),
#     tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
#     tf.keras.layers.Dense(1, activation='sigmoid') # Output layer for binary classification
# ])

# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#               loss='binary_crossentropy', 
#               metrics=['accuracy'])

# model.summary()
