import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_valuation_model():
    # Load dataset
    current_dir = os.path.dirname(__file__)
    data_path = os.path.join(current_dir, 'mumbai_house_data.csv')
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please run data_generator.py first.")
        
    df = pd.read_csv(data_path)
    print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
    
    # Split features and target
    X = df.drop(columns=['Price_Lakhs'])
    y = df['Price_Lakhs']
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    
    # Identify feature types
    categorical_features = ['Locality']
    numerical_features = ['Area_SqFt', 'BHK', 'Bathrooms', 'Property_Age_Years', 'Floor_Num', 'Total_Floors']
    binary_features = ['Parking_Available', 'Swimming_Pool', 'Gym_Available', 'Lift_Available']
    
    # Define preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', StandardScaler(), numerical_features),
            ('bin', 'passthrough', binary_features)
        ]
    )
    
    # Define Random Forest Regressor Model
    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    
    # Create final machine learning pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    
    # Train the pipeline
    print("Training Random Forest Regressor model...")
    pipeline.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = pipeline.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print("\n--- Model Evaluation ---")
    print(f"Mean Absolute Error (MAE): {mae:.2f} Lakhs")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f} Lakhs")
    print(f"R-squared Score (R2): {r2:.4f} ({r2 * 100:.2f}%)")
    
    # Save the pipeline
    model_path = os.path.join(current_dir, 'best_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)
        
    print(f"\nTrained model pipeline saved successfully to {model_path}")
    
    # Output feature importances for visualization later
    print("\nTop Features by Importance:")
    
    # Retrieve feature names after preprocessing
    ohe_categories = pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
    feature_names = list(ohe_categories) + numerical_features + binary_features
    importances = pipeline.named_steps['regressor'].feature_importances_
    
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    print(feature_importance_df.head(10))

if __name__ == '__main__':
    train_valuation_model()
