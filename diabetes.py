import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

def load_and_preprocess_data(data):
    # Basic preprocessing
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def create_correlation_heatmap(data):
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', center=0)
    return plt

def create_feature_importance_plot(model, feature_names):
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    fig = px.bar(importance_df, x='feature', y='importance', 
                 title='Feature Importance',
                 labels={'importance': 'Importance Score', 'feature': 'Features'})
    return fig

def main():
    st.title('Diabetes Prediction System')
    st.write("""
    This application predicts the likelihood of diabetes based on medical predictor variables.
    Please upload a CSV file with the following columns:
    Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome
    """)
    
    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        
        # Data Overview
        st.header('Data Overview')
        st.write('Shape of dataset:', data.shape)
        st.write('First few rows of the dataset:')
        st.dataframe(data.head())
        
        # Data Statistics
        st.header('Statistical Summary')
        st.write(data.describe())
        
        # Visualizations
        st.header('Data Visualizations')
        
        # Correlation Heatmap
        st.subheader('Correlation Heatmap')
        correlation_plot = create_correlation_heatmap(data)
        st.pyplot(correlation_plot.gcf())
        plt.clf()
        
        # Distribution Plots
        st.subheader('Feature Distributions')
        selected_feature = st.selectbox('Select feature to visualize:', data.columns[:-1])
        fig = px.histogram(data, x=selected_feature, color='Outcome',
                          title=f'Distribution of {selected_feature} by Outcome',
                          labels={'Outcome': 'Diabetes Status'})
        st.plotly_chart(fig)
        
        # Model Training
        st.header('Model Training')
        
        X_train_scaled, X_test_scaled, y_train, y_test, scaler = load_and_preprocess_data(data)
        model = train_model(X_train_scaled, y_train)
        
        # Model Performance
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        st.write(f'Model Accuracy: {accuracy:.2%}')
        st.write('Classification Report:')
        st.code(classification_report(y_test, y_pred))
        
        # Feature Importance
        st.subheader('Feature Importance')
        importance_plot = create_feature_importance_plot(model, data.columns[:-1])
        st.plotly_chart(importance_plot)
        
        # Prediction Interface
        st.header('Make Predictions')
        st.write('Enter values to predict diabetes risk:')
        
        # Create input fields for each feature
        input_data = {}
        for feature in data.columns[:-1]:
            input_data[feature] = st.number_input(f'Enter {feature}:', 
                                                value=float(data[feature].mean()),
                                                step=0.1)
        
        if st.button('Predict'):
            # Prepare input data
            input_df = pd.DataFrame([input_data])
            input_scaled = scaler.transform(input_df)
            
            # Make prediction
            prediction = model.predict(input_scaled)
            prediction_proba = model.predict_proba(input_scaled)
            
            st.write('Prediction:', 'Diabetic' if prediction[0] == 1 else 'Non-diabetic')
            st.write(f'Probability of being diabetic: {prediction_proba[0][1]:.2%}')

if __name__ == '__main__':
    main()