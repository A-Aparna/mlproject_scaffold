from src.logger import logging
from src.modules.data_loader import data_loader
from src.modules.data_transformation import data_transformation
from src.modules.model_training_reg import model_training

STAGE_NAME="data_loader"

print(f">>>>>> stage {STAGE_NAME} started <<<<<<")
data_loader_obj = data_loader('data/stud.csv')
data_loader_obj.load()
print(f">>>>>> stage {STAGE_NAME} ended <<<<<<")

STAGE_NAME = "data_transformation"

print(f">>>>>> stage {STAGE_NAME} started <<<<<<")
data_transformation_obj = data_transformation(train_data_path = 'data_artifacts/train_data.csv',
                                            test_data_path = 'data_artifacts/test_data.csv',
                                            target_col='math_score')
train,test = data_transformation_obj.transform()
print(f">>>>>> stage {STAGE_NAME} ended <<<<<<")

STAGE_NAME = "model_training"
print(f">>>>>> stage {STAGE_NAME} started <<<<<<")
model_training_obj = model_training('data_artifacts/train_transformed_data.csv','data_artifacts/test_transformed_data.csv')
training_score,testing_Score = model_training_obj.training()
print("scores for various models",training_score,testing_Score)
print(f">>>>>> stage {STAGE_NAME} ended <<<<<<")