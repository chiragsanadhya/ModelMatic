import streamlit as st
import pandas as pd
from models import RegressionModel, ClassificationModel
from sklearn.model_selection import train_test_split


from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB




# Configure page
st.set_page_config(page_title="Automated Model Training", layout="wide", initial_sidebar_state="expanded")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page:", ["Regression", "Classification"])

if page == "Regression":
    st.title("Regression Model")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Data Preview:")
        st.write(df.head())

        columns_to_remove = st.multiselect("Select columns to remove", df.columns.tolist())
        target = st.text_input("Enter target column name", "")

        if target:
            reg_model = RegressionModel(model=None)  # Replace `None` with a default model if needed
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
            clf_model = ClassificationModel(model=None)  # Replace `None` with a default model if needed
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
