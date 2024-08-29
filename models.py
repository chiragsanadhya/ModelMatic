import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class RegressionModel:
    def __init__(self, model):
        self.model = model
        self.encoder_path = 'encoders.pkl'
    
    def preprocess_data(self, df, columns_to_remove):
        df = df.drop(columns=columns_to_remove, axis=1, errors='ignore')
        columns_to_remove = [column for column in df.columns if df[column].dtype == 'object']
        df = df.drop(columns=columns_to_remove, axis=1, errors='ignore')
        df = df.dropna()
        return df

    def encode_data(self, df, df_test):
        encoders = {}
        for column in df.columns:
            if df[column].dtype == 'object':
                encoder = LabelEncoder()
                df[column] = encoder.fit_transform(df[column].astype(str))
                if column in df_test.columns:
                    df_test[column] = encoder.transform(df_test[column].astype(str))
                encoders[column] = encoder
        with open(self.encoder_path, 'wb') as file:
            pickle.dump(encoders, file)
        return df, df_test

    def prepare_data(self, df, target):
        X = df.drop(target, axis=1, errors='ignore')
        y = df[target]
        return X, y

    def fit_and_predict(self, X_train, y_train, X_test):
        reg = self.model()
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        return y_pred

    def create_output(self, test_id, y_pred, target):
        output = pd.DataFrame({'Id': test_id, target: y_pred})
        return output

    def decode_output(self, output):
        try:
            with open(self.encoder_path, 'rb') as file:
                encoders = pickle.load(file)
            for column, encoder in encoders.items():
                if column in output.columns:
                    output[column] = encoder.inverse_transform(output[column])
        except FileNotFoundError:
            pass
        return output


class ClassificationModel:
    def __init__(self, model):
        self.model = model
        self.encoder_path = 'encoders.pkl'

    def preprocess_data(self, df, columns_to_remove):
        df = df.drop(columns=columns_to_remove, axis=1, errors='ignore')
        columns_to_remove = [column for column in df.columns if df[column].dtype == 'object']
        df = df.drop(columns=columns_to_remove, axis=1, errors='ignore')
        df = df.dropna()
        return df

    def encode_data(self, df, df_test):
        encoders = {}
        for column in df.columns:
            if df[column].dtype == 'object':
                encoder = LabelEncoder()
                df[column] = encoder.fit_transform(df[column].astype(str))
                if column in df_test.columns:
                    df_test[column] = encoder.transform(df_test[column].astype(str))
                encoders[column] = encoder
        with open(self.encoder_path, 'wb') as file:
            pickle.dump(encoders, file)
        return df, df_test

    def prepare_data(self, df, target):
        X = df.drop(target, axis=1, errors='ignore')
        y = df[target]
        return X, y

    def fit_and_predict(self, X_train, y_train, X_test):
        clf = self.model()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        return y_pred

    def create_output(self, df_test, y_pred, target):
        if 'PassengerId' in df_test.columns:
            if len(df_test) == len(y_pred):
                output = pd.DataFrame({'PassengerId': df_test['PassengerId'].values, target: y_pred})
            else:
                output = pd.DataFrame({target: y_pred})
        else:
            output = pd.DataFrame({target: y_pred})
        return output

    def decode_output(self, output):
        try:
            with open(self.encoder_path, 'rb') as file:
                encoders = pickle.load(file)
            for column, encoder in encoders.items():
                if column in output.columns:
                    output[column] = encoder.inverse_transform(output[column])
        except FileNotFoundError:
            pass
        return output
