import os
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

from config import *
import mlflow
from urllib.parse import urlparse

from mlflow.server.auth.client import AuthServiceClient
os.environ['MLFLOW_TRACKING_USERNAME'] = MLFLOW_TRACKING_USERNAME
os.environ['MLFLOW_TRACKING_PASSWORD'] = MLFLOW_TRACKING_PASSWORD

def evaluate(model,X_train,y_train,X_test,y_test):
            model.fit(X_train,y_train)
            y_train_pred = model.predict(X_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test,y_pred)
            train_r2 = r2_score(y_train,y_train_pred)
            test_r2 = r2_score(y_test,y_pred)
            return mse,train_r2,test_r2

def tracking_model_results(model,model_name,train_x,train_y,test_x,test_y):

    print('model_name',model_name)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    with mlflow.start_run():

        (mse, train_r2, test_r2) = evaluate(model,train_x,train_y,test_x,test_y)


        print("  RMSE: %s" % mse)
        print("  MAE: %s" % train_r2)
        print("  R2: %s" % test_r2)

        mlflow.log_param("model_name", model_name)
        mlflow.log_metric("rmse", mse)
        mlflow.log_metric("r2", train_r2)
        mlflow.log_metric("mae", test_r2)

        #predictions = lr.predict(train_x)
        #signature = infer_signature(train_x, predictions)

        ## For Remote server only(DAGShub)

        remote_server_uri=MLFLOW_TRACKING_URI
        mlflow.set_tracking_uri(remote_server_uri)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":
            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path=f"{model_name}",
                )
        else:
            mlflow.sklearn.log_model(model, "model")
    return mse,train_r2,test_r2