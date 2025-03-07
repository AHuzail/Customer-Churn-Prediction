import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import os

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function creates and returns a preprocessing pipeline for data transformation.
        
        Returns:
            ColumnTransformer: A preprocessing object containing pipelines for numerical and categorical columns
            
        Raises:
            CustomException: If any error occurs during pipeline creation
        '''
        try:
            # Define numerical and categorical columns
            numerical_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']
            
            # Categorical columns that need one-hot encoding
            categorical_columns_onehot = [
                'InternetService', 'Contract', 'PaymentMethod',
                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies', 'MultipleLines'
            ]
            
            # Categorical columns that need binary encoding
            categorical_columns_binary = [
                'gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling'
            ]

            logging.info("Creating preprocessing pipelines for numerical and categorical columns")

            # Numerical Pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            # Categorical Pipeline for one-hot encoding with memory optimization
            cat_pipeline_onehot = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(drop='first', sparse=True, handle_unknown='ignore')),
                ]
            )

            # Binary columns are already transformed using map() function
            # Only need pipelines for numerical and one-hot columns
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline_onehot", cat_pipeline_onehot, categorical_columns_onehot)
                ],
                remainder='passthrough'  # This will pass through the already encoded binary columns
            )

            logging.info("Preprocessing pipelines created successfully")
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        Initiates the data transformation process on training and testing datasets.
        
        Args:
            train_path (str): Path to training data CSV
            test_path (str): Path to testing data CSV
            
        Returns:
            tuple: Transformed training array, test array, and preprocessor file path
            
        Raises:
            CustomException: If any error occurs during transformation
        """
        try:
            # Validate file paths
            if not os.path.exists(train_path) or not os.path.exists(test_path):
                raise CustomException("Training or testing file path does not exist", sys)
                
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            # Validate dataframes
            if train_df.empty or test_df.empty:
                raise CustomException("Training or testing data is empty", sys)
                
            logging.info(f"Train DataFrame Head: \n{train_df.head().to_string()}\n")
            logging.info(f"Test DataFrame Head: \n{test_df.head().to_string()}\n")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "Churn"
            
            # Drop customerID as it's not needed
            if "customerID" in train_df.columns:
                train_df = train_df.drop("customerID", axis=1)
                test_df = test_df.drop("customerID", axis=1)

            # Convert TotalCharges to numeric
            train_df['TotalCharges'] = pd.to_numeric(train_df['TotalCharges'], errors='coerce')
            test_df['TotalCharges'] = pd.to_numeric(test_df['TotalCharges'], errors='coerce')

            # Convert binary categorical values
            binary_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
            for col in binary_columns:
                train_df[col] = train_df[col].map({'Yes': 1, 'No': 0, 'Male': 0, 'Female': 1})
                test_df[col] = test_df[col].map({'Yes': 1, 'No': 0, 'Male': 0, 'Female': 1})

            logging.info("Splitting training and test data into features and target")

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            # Validate required columns exist
            required_columns = numerical_columns + categorical_columns_onehot + categorical_columns_binary
            missing_columns = [col for col in required_columns if col not in input_feature_train_df.columns]
            if missing_columns:
                raise CustomException(f"Missing required columns: {missing_columns}", sys)

            logging.info(f"Input feature shapes - Train: {input_feature_train_df.shape}, Test: {input_feature_test_df.shape}")
            logging.info("Applying preprocessing object on training and testing datasets")

            # Apply preprocessing
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Convert sparse matrices to dense if needed
            if isinstance(input_feature_train_arr, (pd.DataFrame, pd.Series)):
                input_feature_train_arr = input_feature_train_arr.to_numpy()
            elif hasattr(input_feature_train_arr, "toarray"):
                input_feature_train_arr = input_feature_train_arr.toarray()
                
            if isinstance(input_feature_test_arr, (pd.DataFrame, pd.Series)):
                input_feature_test_arr = input_feature_test_arr.to_numpy()
            elif hasattr(input_feature_test_arr, "toarray"):
                input_feature_test_arr = input_feature_test_arr.toarray()

            # Combine with target
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Final arrays shape - Train: {train_arr.shape}, Test: {test_arr.shape}")
            logging.info("Saving preprocessing object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)