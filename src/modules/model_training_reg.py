import os
#os.chdir('C:\\Aparna\\mlops')

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from dataclasses import dataclass

from src.logger import logging
from src.helper import get_columns_type, save_object, load_obj
from src.modules.model_evaluation import tracking_model_results, evaluate

STAGE_NAME = "model_training"




@dataclass
class model_training_config:
    model_filepath: str = os.path.join('model_artifacts','model.pkl')
    model_metrics_filepath: str = os.path.join('model_artifacts','model_metrics.txt')


class model_training:

    def __init__(self,train, test):
        self.model_training_config = model_training_config()
        self.train = load_obj(train)
        self.test = load_obj(test)

    def training(self):
        X_train,y_train,X_test,y_test = (self.train[:,:-1],self.train[:,-1],self.test[:,:-1],self.test[:,-1])
        
        models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor()
            }
        
        training_score, testing_score = {}, {}
        
        for model_name,model in models.items():
            logging.info(f"training {model_name} model")
            #mse,train_r2,test_r2 = tracking_model_results(model,model_name,X_train,y_train,X_test,y_test)
            mse,train_r2,test_r2 = evaluate(model,X_train,y_train,X_test,y_test)
            training_score[model_name] = train_r2
            testing_score[model_name] = test_r2

        best_model_score = max(sorted(testing_score.values()))

            ## To get best model name from dict

        best_model_name = list(testing_score.keys())[
                list(testing_score.values()).index(best_model_score)
            ]
        
        best_model = models[best_model_name]
        print(f"best model name: {best_model_name}")


        save_object(self.model_training_config.model_filepath,best_model)
        #tracking_model_results(best_model,best_model_name,X_train,y_train,X_test,y_test)
        predictions = best_model.predict(X_test)
        return training_score, testing_score

if __name__=='__main__':
    logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    model_training_obj = model_training('data_artifacts/train_transformed_data.csv','data_artifacts/test_transformed_data.csv')
    training_score,testing_Score = model_training_obj.training()
    print("scores for various models",training_score,testing_Score)
    logging.info(f">>>>>> stage {STAGE_NAME} ended <<<<<<")