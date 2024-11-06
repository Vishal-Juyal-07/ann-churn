import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load the label encoders, one hot encoder, scaler

with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

with open('one_hot_encoder.pkl', 'rb') as f:
    one_hot_encoder = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)    


# Streamlit app
st.title("Customer Churn prediction")

# User inputs

country = st.selectbox('country',one_hot_encoder.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)
active_member = st.selectbox('Is active member', [0,1])
credit_card = st.selectbox('Has credit card',[0,1])
products_number = st.slider('Number of products', 1, 4)
tenure = st.slider('Tenure',0, 10)
estimated_salary = st.number_input('Estimated Salary')
credit_score = st.number_input('Credit Score')
balance = st.number_input('Balance')
age = st.slider('Age',18,92)


# Prepare the input data
input_data = pd.DataFrame({
    'credit_score':[credit_score],
    'gender':[label_encoder_gender.transform([gender])[0]],
    'age':[age],
    'tenure':[tenure],
    'balance':[balance],
    'products_number':[products_number],
    'credit_card':[credit_card],
    'active_member':[active_member],
    'estimated_salary':[estimated_salary]    
})

# Encoding the country
geo_encoded = one_hot_encoder.transform([[country]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=one_hot_encoder.get_feature_names_out(['country']))

# Combine one hot encoded data with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Make prediction
prediction = model.predict(input_data_scaled)
prediction_probability = prediction[0][0]

# Display the prediction
if prediction_probability>0.5:
    st.write(f'The customer is likely to exit with probability of exit as {prediction_probability}')
else:
    st.write(f'The customer is likely to stay with probability of stay as {1-prediction_probability}')