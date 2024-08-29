import streamlit as st
import pandas as pd
from models import RegressionModel, ClassificationModel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                              AdaBoostRegressor, RandomForestClassifier,
                              GradientBoostingClassifier, AdaBoostClassifier)
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB


st.set_page_config(page_title="ModelMatic", layout="wide", initial_sidebar_state="expanded")


st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page:", ["Home", "Regression", "Classification"])

if page == "Home":
    st.title("Welcome to ModelMatic")
    st.write("""
    **ModelMatic** is your one-stop solution for automated model training and prediction. 

    **How to Use:**
    1. **Upload Your Data**: Start by uploading a CSV file containing your dataset.
    2. **Preprocess Data**: Select columns to remove and specify the target column.
    3. **Choose a Model**: Depending on your needs, select either a regression or classification model.
    4. **Run the Model**: Click on "Run Model" to train and test the selected model.
    5. **View Results**: Check the output predictions displayed on the screen.

    Enjoy effortless machine learning with ModelMatic!
    """)

elif page == "Regression":
    st.title("Regression Model")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Data Preview:")
        st.write(df.head())

        columns_to_remove = st.multiselect("Select columns to remove", df.columns.tolist())
        target = st.text_input("Enter target column name", "")

        if target:
            reg_model = RegressionModel(model=None)
            df = reg_model.preprocess_data(df, columns_to_remove)
            st.write("Preprocessed Data Preview:")
            st.write(df.head())

            X, y = reg_model.prepare_data(df, target)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            X_train, X_test = reg_model.encode_data(X_train, X_test)

            models_dict = {
                "Linear Regression": LinearRegression,
                "K-Nearest Neighbors": KNeighborsRegressor,
                "Support Vector Regression": SVR,
                "Decision Tree": DecisionTreeRegressor,
                "Random Forest": RandomForestRegressor,
                "Gradient Boosting": GradientBoostingRegressor,
                "AdaBoost": AdaBoostRegressor
            }

            model_choice = st.selectbox("Select Regressor", list(models_dict.keys()))

            if st.button("Run Model"):
                model = models_dict[model_choice]
                reg_model.model = model
                y_pred = reg_model.fit_and_predict(X_train, y_train, X_test)
                output = reg_model.create_output(X_test.index, y_pred, target)
                st.write("Final Output:")
                st.write(output.head())

elif page == "Classification":
    st.title("Classification Model")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Data Preview:")
        st.write(df.head())

        target = st.text_input("Enter target column name", "Survived")
        columns_to_remove_input = st.text_input("Enter columns to remove (comma-separated)", "")
        columns_to_remove = [col.strip() for col in columns_to_remove_input.split(",") if col.strip()]

        if target:
            clf_model = ClassificationModel(model=None)
            df = clf_model.preprocess_data(df, columns_to_remove)
            st.write("Preprocessed Data Preview:")
            st.write(df.head())

            X, y = clf_model.prepare_data(df, target)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            X_train, X_test = clf_model.encode_data(X_train, X_test)

            classifiers_dict = {
                "Logistic Regression": LogisticRegression,
                "K-Nearest Neighbors": KNeighborsClassifier,
                "Support Vector Machine": SVC,
                "Decision Tree": DecisionTreeClassifier,
                "Random Forest": RandomForestClassifier,
                "Naive Bayes": GaussianNB,
                "Gradient Boosting": GradientBoostingClassifier,
                "AdaBoost": AdaBoostClassifier
            }

            model_choice = st.selectbox("Select Classifier", list(classifiers_dict.keys()))

            if st.button("Run Model"):
                model = classifiers_dict[model_choice]
                clf_model.model = model
                y_pred = clf_model.fit_and_predict(X_train, y_train, X_test)
                output = clf_model.create_output(df, y_pred, target)
                output = clf_model.decode_output(output)
                st.write("Final Output:")
                st.write(output.head())
