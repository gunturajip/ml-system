from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from joblib import dump
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
from preprocess import preprocess_data
from sklearn.datasets import load_iris
import mlflow
from sklearn.ensemble import RandomForestClassifier # Algoritma dapat disesuaikan
from sklearn.model_selection import GridSearchCV # Optimasi dapat disesuaikan
 
def preprocess_data(data, target_column, save_path, file_path):
    # Menentukan fitur numerik dan kategoris
    numeric_features = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_features = data.select_dtypes(include=['object']).columns.tolist()
    column_names = data.columns
    # Mendapatkan nama kolom tanpa kolom target
    column_names = data.columns.drop(target_column)
 
    # Membuat DataFrame kosong dengan nama kolom
    df_header = pd.DataFrame(columns=column_names)
 
    # Menyimpan nama kolom sebagai header tanpa data
    df_header.to_csv(file_path, index=False)
    print(f"Nama kolom berhasil disimpan ke: {file_path}")
 
    # Pastikan target_column tidak ada di numeric_features atau categorical_features
    if target_column in numeric_features:
        numeric_features.remove(target_column)
    if target_column in categorical_features:
        categorical_features.remove(target_column)
 
    # Pipeline untuk fitur numerik
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
 
    # Pipeline untuk fitur kategoris
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
 
    # Column Transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
 
    # Memisahkan target
    X = data.drop(columns=[target_column])
    y = data[target_column]
 
    # Membagi data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
 
    # Fitting dan transformasi data pada training set
    X_train = preprocessor.fit_transform(X_train)
    # Transformasi data pada testing set
    X_test = preprocessor.transform(X_test)
    # Simpan pipeline
    dump(preprocessor, save_path)
 
    return X_train, X_test, y_train, y_test

# Memuat dataset
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['target'] = iris.target
data.head()
input_example = data[0:5]
 
# Contoh Penggunaan
X_train, X_test, y_train, y_test = preprocess_data(data, 'target', 'preprocessor_pipeline.joblib', 'data.csv')

# Set MLflow Tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
 
# Create a new MLflow Experiment
mlflow.set_experiment("Latihan Model Statis")

with mlflow.start_run():
    mlflow.autolog()
    # Parameter Grid untuk GridSearchCV
    param_grid = {
        'n_estimators': [50, 100, 200, 300, 505, 700],
        'max_depth': [5, 10, 15, 20, 25, 37, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    # Inisialisasi dan Grid Search
    # Initialize RandomForestClassifier
    rf = RandomForestClassifier()
 
    # Perform grid search
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
 
    # Mendapatkan Parameter dan Model Terbaik
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
 
    # Log best parameters
    mlflow.log_params(best_params)
 
    # Train the best model on the training set
    best_model.fit(X_train, y_train)
 
    # Logging model
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="model",
        input_example=input_example
    )
 
    # Evaluate the model on the test set and log accuracy
    accuracy = best_model.score(X_test, y_test)