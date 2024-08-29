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
from sklearn.model_selection import train_test_split

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
            if column in df_test.columns:
                df_test[column] = encoder.transform(df_test[column].astype(str))
            encoders[column] = encoder
    with open('encoders.pkl', 'wb') as file:
        pickle.dump(encoders, file)
    return df, df_test

def prepare_data(df, target):
    X = df.drop(target, axis=1, errors='ignore')
    y = df[target]
    return X, y

def fit_and_predict(X_train, y_train, X_test, classifier):
    clf = classifier()
    clf.fit(X_train, y_train)
    y_test = clf.predict(X_test)
    return y_test

def create_output(df_test, y_test, target):
    if isinstance(df_test, pd.DataFrame) and 'PassengerId' in df_test.columns:
        if len(df_test) == len(y_test):
            output = pd.DataFrame({'PassengerId': df_test['PassengerId'].values, target: y_test})
        else:
            output = pd.DataFrame({target: y_test})
    else:
        output = pd.DataFrame({target: y_test})
    return output

def decode_output(output):
    try:
        with open('encoders.pkl', 'rb') as file:
            encoders = pickle.load(file)
        for column, encoder in encoders.items():
            if column in output.columns:
                output[column] = encoder.inverse_transform(output[column])
    except FileNotFoundError:
        st.warning("Encoders file not found. Skipping decoding.")
    return output


st.title('Prediction Model (Classification problem)')

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.write("Data Preview:")
    st.write(df.head())

    target = st.text_input("Enter the target column name", "Survived")
    columns_to_remove_input = st.text_input("Enter columns to remove (comma-separated)", "")
    columns_to_remove = [col.strip() for col in columns_to_remove_input.split(",") if col.strip()]


    df = preprocess_data(df, columns_to_remove)

    st.write("Preprocessed Data Preview:")
    st.write(df.head())


    split_ratio = st.slider("Select train-test split ratio (percentage for training data)", 1, 99, 80)
    X, y = prepare_data(df, target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(100 - split_ratio) / 100, random_state=42)


    X_train, X_test = encode_data(X_train, X_test)

    st.write("Encoded Training Data Preview:")
    st.write(X_train.head())

    st.write("Encoded Test Data Preview:")
    st.write(X_test.head())

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
        y_pred = fit_and_predict(X_train, y_train, X_test, classifier)


        test_id = df.get('PassengerId', pd.DataFrame()) # Get test IDs if available
        output = create_output(test_id, y_pred, target)


        output = decode_output(output)

        st.write("Final Output:")
        st.write(output.head())
