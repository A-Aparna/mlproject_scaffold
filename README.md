# Machine Learning Pipeline Implementation
This repo concentrates on:
   - Creating a project structure
   - Training module
   - Inference by exposing the API
   - Tracking the delta in data by versioning the data
   - Saving the experimentations of various model performances
   - CI/CD pipeline
   - Deploying at an endpoint on cloud

## How to run
  `uvicorn app:app`
  
## Tools used:
  - FastAPI
  - DVC
  - MLflow
  - AWS

## What each file means?
  - .github/workflows/aws.yaml
      -   Creates CI/CD/CD pipeline from github actions
      -   Builds tags and pushes images to the ECR
      -   Deploys ECS task definition
        
  - src/modules/data_loader.py
      - Downloads the data set from the 3rd party location (google drive in this case)
      - train splits the data
      - write to the artifacts folder. These files are tracked using DVC
        
  - src/modules/data_transformation.py
      - Feature engineering the split data
      - Save the transformed data to the artifacts folder. These files are tracked using DVC
      - Save the transormation object to be used by inference in the model_Artifacts folder
        
  - src/modules/evaluation.py
      - Calcualtes the training and test error for regression model
      - Uplaods the data to MLflow to track the experimental results
        
  - src/modules/model_training_reg.py
      - Trains the data on various models
      - Selects the best model based on the error
      - Writes the selected model to the model_artifacts folder which is also tracked by DVC
   
  - src/pipeline/helper.py
      - Helper functions for support the pipeline running

  - src/pipeline/logger.py
      - Formatting the way log message is displayed
      - Saving log files
  
  - app.py
      - Creating an app using FastAPI
      - Exposing 3 end points- home, train and predict
      - Train uses get to run the pipeline
      - Predict uses Post to take input from UX for different features and returns predicted values
        
  - config.py
      - Static file with login credentials
        
  - Dockerfile
      - creates a python image
      - runs requirements.txt
      - runs the uvicorn to run the app in app.py

  - dvc.yaml
      - This yaml file configures the pipeline
      -  tracks the stages, inputs and output  of the stages.
     
  - Main.py
      - Standone file to run the training without using fastAPI to deploy the app
    
  - post_data_validation.py
      - Uses Pydantic base model to validate the inputs of the inference from FastAPI end point
    
  - requirements.txt
      - Liste of Libraris to be installed in the docker container
    
  - setup.py
    
