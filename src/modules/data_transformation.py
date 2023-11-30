import os
#os.chdir('C:\\Aparna\\mlops')
import pandas as pd
import numpy as np
from dataclasses import dataclass

from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.logger import logging
from src.helper import get_columns_type, save_object
from src.modules.model_training_reg import model_training

STAGE_NAME = "data_transformation"
@dataclass
class data_transformation_config:
    transformation_filepath = os.path.join('model_artifacts','transformation.pkl')
    train_transformed_data_path = os.path.join('data_artifacts','train_transformed_data.csv')
    test_transformed_data_path = os.path.join('data_artifacts','test_transformed_data.csv')

    
class data_transformation:
    def __init__(self,train_data_path,test_data_path, target_col):
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.data_transformation_config = data_transformation_config()
        self.train_df = None
        self.test_df = None
        self.numerical_cols = None
        self.categorical_cols = None
        self.target_col = target_col
        self.preprocessor = None
        


    def transform(self):

        self.load_data()
        self.define_transformer()
        self.transform_data()
        return self.train_df, self.test_df


    def load_data(self):

        self.train_df = pd.read_csv(self.train_data_path)
        self.test_df = pd.read_csv(self.test_data_path)
        logging.info(f"train_df shape: {self.train_df.shape}")
        logging.info(f"test_df shape: {self.test_df.shape}")


    def define_transformer(self):
        """
        This function will transform the data
        """
        categorical_cols = get_columns_type(self.train_df,['object'])
        numeric_cols = get_columns_type(self.train_df,['int','float'])  
        self.categorical_cols = [i for i in categorical_cols if i!=self.target_col]
        self.numerical_cols = [i for i in numeric_cols if i!=self.target_col]

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore')),
            ("scaler",StandardScaler(with_mean=False))])
        
        logging.info(f"numeric_cols: {self.numerical_cols}")
        logging.info(f"categorical_cols: {self.categorical_cols}")

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numerical_cols),
                ('cat', categorical_transformer, self.categorical_cols)])


    def transform_data(self):
        """
        This function will transform the data
        """
        train_df_feature = self.train_df.drop(self.target_col,axis=1)
        test_df_feature = self.test_df.drop(self.target_col,axis=1)
        train_df_target = self.train_df[self.target_col]
        test_df_target =  self.test_df[self.target_col]

        logging.info("transforming train data")

        train_prep = self.preprocessor.fit_transform(train_df_feature)

        logging.info("transforming test data")

        test_prep = self.preprocessor.transform(test_df_feature)

        self.train_df = np.c_[train_prep,train_df_target]
        self.test_df = np.c_[test_prep,test_df_target]  

        logging.info(f"train_df shape: {self.train_df.shape}")
        logging.info(f"test_df shape: {self.test_df.shape}")


        logging.info(f"saving transformation at {self.data_transformation_config.transformation_filepath}")
        print(f"saving transformation at {self.data_transformation_config.transformation_filepath}")
        save_object(self.data_transformation_config.transformation_filepath,self.preprocessor)
        save_object(self.data_transformation_config.train_transformed_data_path,self.train_df)
        save_object(self.data_transformation_config.test_transformed_data_path,self.test_df)


if __name__=='__main__':

    logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    data_transformation_obj = data_transformation(train_data_path = 'data_artifacts/train_data.csv',
                                                test_data_path = 'data_artifacts/test_data.csv',
                                                target_col='math_score')
    train,test = data_transformation_obj.transform()
    logging.info(f">>>>>> stage {STAGE_NAME} ended <<<<<<")
    #model_training_obj = model_training(train,test)
    #training_score,testing_Score = model_training_obj.training()
    #print("scores for various models",training_score,testing_Score)
        