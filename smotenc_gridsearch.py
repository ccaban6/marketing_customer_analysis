from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import SMOTENC
import pandas as pd
import numpy as np

# Features for campaign response
numerical_features = ['Age', 'Income', 'Total_Children', 'Education_Numeric',
                     'R_Score', 'F_Score', 'M_Score', 'NumWebVisitsMonth',
                     'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts',
                     'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases',
                     'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases',
                     'Prefers_Web', 'Deal_Seeker', 'Marital_Divorced', 'Marital_Married', 
                     'Marital_Single', 'Marital_Together', 'Marital_Widow', 
                     'Country_AUS', 'Country_CA', 'Country_GER', 'Country_IND', 
                     'Country_ME', 'Country_SA', 'Country_SP', 'Country_US']
categorical_features = ['Primary_Product', 'Customer_Persona']

# IMPORTANT: Preserve column order
all_columns = numerical_features + categorical_features

class SMOTENCPreprocessor(BaseEstimator, TransformerMixin):
    """Custom transformer that handles SMOTENC preprocessing"""
    
    def __init__(self, numerical_features, categorical_features, random_state=42):
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.random_state = random_state
        self.all_columns = numerical_features + categorical_features
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.smotenc = None
        self.categorical_indices = [self.all_columns.index(col) for col in categorical_features]
        
    def fit(self, X, y=None):
        # Order columns
        X_ordered = X[self.all_columns].copy()
        
        # Fit label encoders for categorical features
        for col in self.categorical_features:
            le = LabelEncoder()
            le.fit(X_ordered[col])
            self.label_encoders[col] = le
            
        # Fit scaler for numerical features
        self.scaler.fit(X_ordered[self.numerical_features])
        
        # Initialize SMOTENC
        self.smotenc = SMOTENC(categorical_features=self.categorical_indices, 
                              random_state=self.random_state)
        
        return self
    
    def transform(self, X):
        # Order columns
        X_ordered = X[self.all_columns].copy()
        
        # Label encode categorical features
        for col in self.categorical_features:
            le = self.label_encoders[col]
            # Handle unseen categories by mapping them to -1
            X_ordered[col] = X_ordered[col].map(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
        
        # Scale numerical features
        X_ordered[self.numerical_features] = self.scaler.transform(X_ordered[self.numerical_features])
        
        # One-hot encode categorical features
        X_final = pd.get_dummies(X_ordered, columns=self.categorical_features)
        
        return X_final
    
    def fit_resample(self, X, y):
        # Order columns
        X_ordered = X[self.all_columns].copy()
        
        # Label encode categorical features
        for col in self.categorical_features:
            le = LabelEncoder()
            X_ordered[col] = le.fit_transform(X_ordered[col])
            self.label_encoders[col] = le
            
        # Scale numerical features
        X_ordered[self.numerical_features] = self.scaler.fit_transform(X_ordered[self.numerical_features])
        
        # Apply SMOTENC
        self.smotenc = SMOTENC(categorical_features=self.categorical_indices, 
                              random_state=self.random_state)
        X_resampled, y_resampled = self.smotenc.fit_resample(X_ordered.values, y)
        
        # Convert back to DataFrame
        X_resampled_df = pd.DataFrame(X_resampled, columns=self.all_columns)
        
        # Decode categorical features
        for col in self.categorical_features:
            X_resampled_df[col] = self.label_encoders[col].inverse_transform(
                X_resampled_df[col].astype(int)
            )
        
        # One-hot encode
        X_final = pd.get_dummies(X_resampled_df, columns=self.categorical_features)
        
        # Store training columns for later alignment
        self.training_columns = X_final.columns.tolist()
        
        return X_final, y_resampled
    
class SMOTENCGridSearchCV:
    """Custom GridSearchCV implementation for SMOTENC"""
    
    def __init__(self, estimator, param_grid, scoring='f1', cv=5, n_jobs=-1, verbose=1):
        self.estimator = estimator
        self.param_grid = param_grid
        self.scoring = scoring
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.best_estimator_ = None
        self.best_params_ = None
        self.best_score_ = None
        
    def fit(self, X, y):
        # Use GridSearchCV but with custom preprocessing
        from sklearn.model_selection import GridSearchCV
        
        # Apply SMOTENC preprocessing
        preprocessor = SMOTENCPreprocessor(numerical_features, categorical_features)
        X_resampled, y_resampled = preprocessor.fit_resample(X, y)
        
        # Store preprocessor for later use
        self.preprocessor = preprocessor
        
        # Create GridSearchCV with resampled data
        grid_search = GridSearchCV(
            estimator=self.estimator,
            param_grid=self.param_grid,
            scoring=self.scoring,
            cv=self.cv,
            n_jobs=self.n_jobs,
            verbose=self.verbose
        )
        
        # Fit on resampled data
        grid_search.fit(X_resampled, y_resampled)
        
        # Store best results
        self.best_estimator_ = grid_search.best_estimator_
        self.best_params_ = grid_search.best_params_
        self.best_score_ = grid_search.best_score_
        self.grid_search = grid_search
        
        return self
    
    def predict(self, X):
        # Transform test data to match training data
        X_transformed = self.preprocessor.transform(X)
        
        # Ensure test set has same columns as training set
        missing_cols = set(self.preprocessor.training_columns) - set(X_transformed.columns)
        for col in missing_cols:
            X_transformed[col] = 0
            
        # Remove extra columns
        extra_cols = set(X_transformed.columns) - set(self.preprocessor.training_columns)
        X_transformed = X_transformed.drop(columns=extra_cols)
        
        # Reorder columns to match training data
        X_transformed = X_transformed[self.preprocessor.training_columns]
        
        return self.best_estimator_.predict(X_transformed)
    
    def predict_proba(self, X):
        # Transform test data to match training data
        X_transformed = self.preprocessor.transform(X)
        
        # Ensure test set has same columns as training set
        missing_cols = set(self.preprocessor.training_columns) - set(X_transformed.columns)
        for col in missing_cols:
            X_transformed[col] = 0
            
        # Remove extra columns
        extra_cols = set(X_transformed.columns) - set(self.preprocessor.training_columns)
        X_transformed = X_transformed.drop(columns=extra_cols)
        
        # Reorder columns to match training data
        X_transformed = X_transformed[self.preprocessor.training_columns]
        
        return self.best_estimator_.predict_proba(X_transformed)
