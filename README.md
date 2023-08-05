# Random Forest for Binary Classification
 Machine Learning for Binary Classification Models - Ready Tensor


## Project Description

A short description of the project.

## Project Structure

```txt
your_chosen_project_name/
├── examples/
├── model_inputs_outputs/
│   ├── inputs/
│   │   ├── data/
│   │   │   ├── testing/
│   │   │   └── training/
│   │   └── schema/
│   ├── model/
│   │   └── artifacts/
│   └── outputs/
│       ├── errors/
│       ├── hpt_outputs/
│       └── predictions/
├── requirements/
│   ├── requirements.txt
│   └── requirements_text.txt "not implemented yet"
├── src/
│   ├── config/
│   ├── data_models/
│   ├── hyperparameter_tuning/
│   ├── prediction/
│   ├── preprocessing/
│   ├── schema/
│   └── xai/
├── tests/ "not implemented yet"
│   ├── integration_tests/
│   ├── performance_tests/
│   └── unit_tests/
│       ├── (mirrors /src structure)
│       └── ...
├── tmp/
├── .gitignore
├── LICENSE
└── README.md
```

- **`/examples`**: This directory contains all files you want to use as examples. Typically, these would be small data files that can be used as examples for the ML project.
- **`/model_inputs_outputs`**: This directory contains files that are either inputs to, or outputs from, the model. This directory is further divided into:
  - **`/inputs`**: This directory contains all the input files for your project, including the `data` and `schema` files. The `data` is further divided into `testing` and `training` subsets.
  - **`/model/artifacts`**: This directory is used to store the model artifacts, such as trained models and their parameters.
  - **`/outputs`**: The outputs directory contains sub-directories for error logs, and hyperparameter tuning outputs, and prediction results. Note that model artifacts should not be saved in this directory. Instead, they should be saved in the `/model/artifacts` directory.
- **`requirements`**: This directory contains the requirements files. You may create multiple requirements files for different purposes, such as `requirements.txt` for the main code in the `src` directory. You may optionally have another requirements file to use for linting and style checks.
- **`/src`**: This directory holds the source code for the project. It is further divided into various subdirectories such as `config` for configuration files, `data_models` for data models for input validation, `hyperparameter_tuning` for hyperparameter-tuning (HPT) related files, `prediction` for prediction model scripts, `preprocessing` for data preprocessing scripts, `schema` for schema handler scripts, and `xai` for explainable AI scripts.
  - **`serve.py`**: This script is used to serve the model as a REST API using **FastAPI**. It loads the artifacts and creates a FastAPI server to serve the model. It provides 3 endpoints: `/ping`, `/infer`, and `/explain`. The `/ping` endpoint is used to check if the server is running. The `/infer` endpoint is used to make predictions. The `/explain` endpoint is used to get local explanations for the predictions.
  - **`serve_utils.py`**: This script contains utility functions used by the `serve.py` script.
  - **`logger.py`**: This script contains the logger configuration using **logging** module.
  - **`train.py`**: This script is used to train the model. It loads the data, preprocesses it, trains the model, and saves the artifacts in the path `./model_inputs_outputs/model/artifacts/`. It also saves a SHAP explainer object in the path `./model/artifacts/`. When the train task is run with a flag to perform hyperparameter tuning, it also saves the hyperparameter tuning results in the path `./model_inputs_outputs/outputs/hpt_outputs/`.
  - **`predict.py`**: This script is used to run batch predictions using the trained model. It loads the artifacts and creates and saves the predictions in a file called `predictions.csv` in the path `./model_inputs_outputs/outputs/predictions/`.
  - **`utils.py`**: This script contains utility functions used by the other scripts.
- **`/tests`**: This directory contains all the tests for the project. It contains sub-directories for specific types of tests such as unit tests, integration tests, and performance tests. For unit tests, the directory structure should mirror the `/src` directory structure. For example, if you have a `preprocessing` folder in the `/src` directory, then tests corresponding to the scripts under `src/preprocessing` should be placed in the `/tests/unit_tests/preprocessing` directory.
- **`/tmp`**: This directory is used for storing temporary files which are not necessary to commit to the repository.
- **`.gitignore`**: This file specifies the files and folders that should be ignored by Git.
- **`LICENSE`**: This file contains an MIT license for the project. You can change this to any other license you want.
- **`README.md`**: This file contains the documentation for the project, explaining how to set it up and use it.

## Usage
In this section we cover the following:
- How to prepare your data for training and inference
- How to run the model implementation locally (without Docker)
- How to run the model implementation with Docker
- How to use the inference service (with or without Docker)
### Preparing your data
- If you plan to run this model implementation on your own binary classification dataset, you will need your training and testing data in a CSV format. Also, you will need to create a schema file as per the Ready Tensor specifications. The schema is in JSON format, and it's easy to create. You can use the example schema file provided in the `examples` directory as a template.
### To run locally (without Docker)
- Create your virtual environment and install dependencies listed in `requirements.txt` which is inside the `requirements` directory.
- Move the three example files (`titanic_schema.json`, `titanic_train.csv` and `titanic_test.csv`) in the `examples` directory into the `./model_inputs_outputs/inputs/schema`, `./model_inputs_outputs/inputs/data/training` and `./model_inputs_outputs/inputs/data/testing` folders, respectively (or alternatively, place your custom dataset files in the same locations).
- Run the script `src/train.py` to train the random forest classifier model. This will save the model artifacts, including the preprocessing pipeline and label encoder, in the path `./model_inputs_outputs/model/artifacts/`. If you want to run with hyperparameter tuning then include the `-t` flag. This will also save the hyperparameter tuning results in the path `./model_inputs_outputs/outputs/hpt_outputs/`.
- Run the script `src/predict.py` to run batch predictions using the trained model. This script will load the artifacts and create and save the predictions in a file called `predictions.csv` in the path `./model_inputs_outputs/outputs/predictions/`.
- Run the script `src/serve.py` to start the inference service, which can be queried using the `/ping`, `/infer` and `/explain` endpoints. The service runs on port 8080.
### To run with Docker
1. Set up a bind mount on host machine: It needs to mirror the structure of the `model_inputs_outputs` directory. Place the train data file in the `model_inputs_outputs/inputs/data/training` directory, the test data file in the `model_inputs_outputs/inputs/data/testing` directory, and the schema file in the `model_inputs_outputs/inputs/schema` directory.
2. Build the image. You can use the following command: <br/>
   `docker build -t classifier_img .` <br/>
   Here `classifier_img` is the name given to the container (you can choose any name).
3. Note the following before running the container for train, batch prediction or inference service:
   - The train, batch predictions tasks and inference service tasks require a bind mount to be mounted to the path `/opt/model_inputs_outputs/` inside the container. You can use the `-v` flag to specify the bind mount.
   - When you run the train or batch prediction tasks, the container will exit by itself after the task is complete. When the inference service task is run, the container will keep running until you stop or kill it.
   - When you run training task on the container, the container will save the trained model artifacts in the specified path in the bind mount. This persists the artifacts even after the container is stopped or killed.
   - When you run the batch prediction or inference service tasks, the container will load the trained model artifacts from the same location in the bind mount. If the artifacts are not present, the container will exit with an error.
   - The inference service runs on the container's port **8080**. Use the `-p` flag to map a port on local host to the port 8080 in the container.
   - Container runs as user 1000. Provide appropriate read-write permissions to user 1000 for the bind mount. Please follow the principle of least privilege when setting permissions. The following permissions are required:
     - Read access to the `inputs` directory in the bind mount. Write or execute access is not required.
     - Read-write access to the `outputs` directory and `model` directories. Execute access is not required.
4. You can run training with or without hyperparameter tuning:
   - To run training without hyperparameter tuning (i.e. using default hyperparameters), run the container with the following command container: <br/>
     `docker run -v <path_to_mount_on_host>/model_inputs_outputs:/opt/model_inputs_outputs classifier_img train` <br/>
     where `classifier_img` is the name of the container. This will train the model and save the artifacts in the `model_inputs_outputs/model/artifacts` directory in the bind mount.
   - To run training with hyperparameter tuning, issue the command: <br/>
     `docker run -v <path_to_mount_on_host>/model_inputs_outputs:/opt/model_inputs_outputs classifier_img train -t` <br/>
     This will tune hyperparameters,and used the tuned hyperparameters to train the model and save the artifacts in the `model_inputs_outputs/model/artifacts` directory in the bind mount. It will also save the hyperparameter tuning results in the `model_inputs_outputs/outputs/hpt_outputs` directory in the bind mount.
5. To run batch predictions, place the prediction data file in the `model_inputs_outputs/inputs/data/testing` directory in the bind mount. Then issue the command: <br/>
   `docker run -v <path_to_mount_on_host>/model_inputs_outputs:/opt/model_inputs_outputs classifier_img predict` <br/>
   This will load the artifacts and create and save the predictions in a file called `predictions.csv` in the path `model_inputs_outputs/outputs/predictions/` in the bind mount.
6. To run the inference service, issue the following command on the running container: <br/>
   `docker run -p 8080:8080 -v <path_to_mount_on_host>/model_inputs_outputs:/opt/model_inputs_outputs classifier_img serve` <br/>
   This starts the service on port 8080. You can query the service using the `/ping`, `/infer` and `/explain` endpoints. More information on the requests/responses on the endpoints is provided below.
### Using the Inference Service
#### Getting Predictions
To get predictions for a single sample, use the following command:
```bash
curl -X POST -H "Content-Type: application/json" -d '{
  {
    "instances": [
        {
            "PassengerId": "879",
            "Pclass": 3,
            "Name": "Laleff, Mr. Kristo",
            "Sex": "male",
            "Age": None,
            "SibSp": 0,
            "Parch": 0,
            "Ticket": "349217",
            "Fare": 7.8958,
            "Cabin": None,
            "Embarked": "S"
        }
    ]
}' http://localhost:8080/infer
```
The key `instances` contains a list of objects, each of which is a sample for which the prediction is requested. The server will respond with a JSON object containing the predicted probabilities for each input record:
```json
{
  "status": "success",
  "message": "",
  "timestamp": "<timestamp>",
  "requestId": "<uniquely generated id>",
  "targetClasses": ["0", "1"],
  "targetDescription": "A binary variable indicating whether or not the passenger survived (0 = No, 1 = Yes).",
  "predictions": [
    {
      "sampleId": "879",
      "predictedClass": "0",
      "predictedProbabilities": [0.97548, 0.02452]
    }
  ]
}
```
#### Getting predictions and local explanations
To get predictions and explanations for a single sample, use the following request to send to `/explain` endpoint (same structure as data for the `/infer` endpoint):
```bash
curl -X POST -H "Content-Type: application/json" -d '{
  {
    "instances": [
        {
            "PassengerId": "879",
            "Pclass": 3,
            "Name": "Laleff, Mr. Kristo",
            "Sex": "male",
            "Age": None,
            "SibSp": 0,
            "Parch": 0,
            "Ticket": "349217",
            "Fare": 7.8958,
            "Cabin": None,
            "Embarked": "S"
        }
    ]
}' http://localhost:8080/explain
```
The server will respond with a JSON object containing the predicted probabilities and locations for each input record:
```json
{
  "status": "success",
  "message": "",
  "timestamp": "2023-05-22T10:51:45.860800",
  "requestId": "0ed3d0b76d",
  "targetClasses": ["0", "1"],
  "targetDescription": "A binary variable indicating whether or not the passenger survived (0 = No, 1 = Yes).",
  "predictions": [
    {
      "sampleId": "879",
      "predictedClass": "0",
      "predictedProbabilities": [0.92107, 0.07893],
      "explanation": {
        "baseline": [0.57775, 0.42225],
        "featureScores": {
          "Age_na": [0.05389, -0.05389],
          "Age": [0.02582, -0.02582],
          "SibSp": [-0.00469, 0.00469],
          "Parch": [0.00706, -0.00706],
          "Fare": [0.05561, -0.05561],
          "Embarked_S": [0.01582, -0.01582],
          "Embarked_C": [0.00393, -0.00393],
          "Embarked_Q": [0.00657, -0.00657],
          "Pclass_3": [0.0179, -0.0179],
          "Pclass_1": [0.02394, -0.02394],
          "Sex_male": [0.13747, -0.13747]
        }
      }
    }
  ],
  "explanationMethod": "Shap"
}
```

## Requirements

```python
pip install -r requirements/requirements.txt
```

## License
This project is provided under the MIT License. Please see the [LICENSE](LICENSE) file for more information.

## Contact Information
email: mahmoud.hesham.saadd@gmail.com


