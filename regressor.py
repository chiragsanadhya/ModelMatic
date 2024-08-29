import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
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
            df_test[column] = encoder.transform(df_test[column].astype(str))
            encoders[column] = encoder
    with open('encoders.pkl', 'wb') as file:
        pickle.dump(encoders, file)
    return df, df_test

def prepare_data(df, target):
    X = df.drop(target, axis=1, errors='ignore')
    y = df[target]
    return X, y

def fit_and_predict(X_train, y_train, X_test, regressor):
    reg = regressor()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    return y_pred

def create_output(test_id, y_pred, target):
    output = pd.DataFrame({'Id': test_id, target: y_pred})
    return output

def decode_output(output):
    with open('encoders.pkl', 'rb') as file:
        encoders = pickle.load(file)
    for column, encoder in encoders.items():
        if column in output.columns:
            output[column] = encoder.inverse_transform(output[column])
    return output

# Streamlit app
st.title('Regression Model')

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.write("Data Preview:")
    st.write(df.head())

    st.write("Select columns to remove (3 columns per line):")

    columns = df.columns.tolist()

    # Create three columns for checkboxes
    col1, col2, col3 = st.columns(3)

    # Display checkboxes in the columns
    columns_to_remove = []
    with col1:
        for column in columns[:len(columns)//3]:
            if st.checkbox(column, key=column):
                columns_to_remove.append(column)
    
    with col2:
        for column in columns[len(columns)//3:2*len(columns)//3]:
            if st.checkbox(column, key=column):
                columns_to_remove.append(column)
    
    with col3:
        for column in columns[2*len(columns)//3:]:
            if st.checkbox(column, key=column):
                columns_to_remove.append(column)

    # Display selected columns to remove
    st.write("Columns to be removed:")
    st.write(columns_to_remove)

    # Preprocess data
    df = preprocess_data(df, columns_to_remove)

    st.write("Preprocessed Data Preview:")
    st.write(df.head())

    # Get the target column
    target = st.text_input("Enter target column", "")

    if target and target in df.columns:
        X, y = prepare_data(df, target)

        # Split the data
        train_size = st.slider("Training data ratio (%)", min_value=1, max_value=99, value=80)
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size/100, random_state=42)

        regressors = {
            "Linear Regression": LinearRegression,
            "K-Nearest Neighbors": KNeighborsRegressor,
            "Support Vector Regression": SVR,
            "Decision Tree": DecisionTreeRegressor,
            "Random Forest": RandomForestRegressor,
            "Gradient Boosting": GradientBoostingRegressor,
            "AdaBoost": AdaBoostRegressor
        }

        regressor_choice = st.selectbox("Select Regressor", list(regressors.keys()))

        if st.button("Run Model"):
            regressor = regressors[regressor_choice]
            y_pred = fit_and_predict(X_train, y_train, X_test, regressor)

            # Ensure test_id is derived from X_test
            test_id = X_test.index
            output = create_output(test_id, y_pred, target)

            st.write("Final Output:")
            st.write(output.head())
    else:
        st.write("Please enter a valid target column name.")
