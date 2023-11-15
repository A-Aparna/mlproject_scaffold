import os
os.chdir('C:\\Aparna\\mlops')

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from dataclasses import dataclass

from src.logger import logging
from src.helper import get_columns_type, save_object


def evaluate(model,X_train,y_train,X_test,y_test):
            model.fit(X_train,y_train)
            y_train_pred = model.predict(X_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test,y_pred)
            train_r2 = r2_score(y_train,y_train_pred)
            test_r2 = r2_score(y_test,y_pred)
            return train_r2,test_r2


@dataclass
class model_training_config:
    model_filepath: str = os.path.join('model_artifacts','model.pkl')
    model_metrics_filepath: str = os.path.join('model_artifacts','model_metrics.txt')


class model_training:

    def __init__(self,train, test):
        self.model_training_config = model_training_config()
        self.train = train
        self.test = test

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
            train_r2,test_r2 = evaluate(model,X_train,y_train,X_test,y_test)
            training_score[model_name] = train_r2
            testing_score[model_name] = test_r2

        best_model_score = max(sorted(testing_score.values()))

            ## To get best model name from dict

        best_model_name = list(testing_score.keys())[
                list(testing_score.values()).index(best_model_score)
            ]
        
        best_model = models[best_model_name]
        logging.info(f"best model name: {best_model_name}")


        save_object(self.model_training_config.model_filepath,best_model)

        predictions = best_model.predict(X_test)
        return training_score, testing_score

if __name__=='__main__':
    
    model_training_obj = model_training('data_artifacts/train_data.csv','data_artifacts/test_data.csv')
    model_training_obj.training()