import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

def preprocess_data(df, columns_to_remove):
    df = df.drop(columns=columns_to_remove, axis=1, errors='ignore')
    columns_to_remove = [column for column in df.columns if df[column].dtype == 'object']
    df = df.drop(columns=columns_to_remove, axis=1, errors='ignore')
    df = df.dropna()
    return df

def encode_data(df, df_test):
    encoders = {}
    for column in df.columns:
        if df[column].dtype == 'object':
            encoder = LabelEncoder()
            df[column] = encoder.fit_transform(df[column].astype(str))
            df_test[column] = encoder.transform(df_test[column].astype(str))
            encoders[column] = encoder
    with open('encoders.pkl', 'wb') as file:
        pickle.dump(encoders, file)
    return df, df_test

def prepare_data(df, df_test, target):
    X_train = df.drop(target, axis=1, errors='ignore')
    y_train = df[target]
    X_test = df_test
    return X_train, y_train, X_test

def fit_and_predict(X_train, y_train, X_test, classifier):
    clf = classifier()
    clf.fit(X_train, y_train)
    y_test = clf.predict(X_test)
    return y_test

def create_output(test_id, y_test, target):
    output = pd.DataFrame({'PassengerId': test_id.values, target: y_test})
    return output

def decode_output(output):
    with open('encoders.pkl', 'rb') as file:
        encoders = pickle.load(file)
    for column, encoder in encoders.items():
        if column in output.columns:
            output[column] = encoder.inverse_transform(output[column])
    return output


st.title('Prediction Model (Classification problem)')

uploaded_train = st.file_uploader("Upload training CSV file", type=["csv"])
uploaded_test = st.file_uploader("Upload test CSV file", type=["csv"])

if uploaded_train and uploaded_test:
    df = pd.read_csv(uploaded_train)
    df_test = pd.read_csv(uploaded_test)

    st.write("Training Data Preview:")
    st.write(df.head())

    st.write("Test Data Preview:")
    st.write(df_test.head())

    columns_to_remove_input = st.text_input("Enter columns to remove (comma-separated)", "")
    columns_to_remove = [col.strip() for col in columns_to_remove_input.split(",") if col.strip()]


    df = preprocess_data(df, columns_to_remove)
    df_test = preprocess_data(df_test, columns_to_remove)

    st.write("Preprocessed Training Data Preview:")
    st.write(df.head())

    st.write("Preprocessed Test Data Preview:")
    st.write(df_test.head())


    df, df_test = encode_data(df, df_test)

    st.write("Encoded Training Data Preview:")
    st.write(df.head())

    st.write("Encoded Test Data Preview:")
    st.write(df_test.head())


    target = "Survived"
    X_train, y_train, X_test = prepare_data(df, df_test, target)


    classifiers = {
        "Logistic Regression": LogisticRegression,
        "K-Nearest Neighbors": KNeighborsClassifier,
        "Support Vector Machine": SVC,
        "Decision Tree": DecisionTreeClassifier,
        "Random Forest": RandomForestClassifier,
        "Naive Bayes": GaussianNB,
        "Gradient Boosting": GradientBoostingClassifier,
        "AdaBoost": AdaBoostClassifier
    }

    classifier_choice = st.selectbox("Select Classifier", list(classifiers.keys()))

    if st.button("Run Model"):
        classifier = classifiers[classifier_choice]
        y_test = fit_and_predict(X_train, y_train, X_test, classifier)


        test_id = df_test['PassengerId']
        output = create_output(test_id, y_test, target)


        output = decode_output(output)

        st.write("Final Output:")
        st.write(output.head())
