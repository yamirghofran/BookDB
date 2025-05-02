import pandas as pd
import dask.dataframe as dd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import boto3
import yaml
import os

class PipelineStep(object):
    def __init__(self, config):
        self.config = config

    def run(self, inputs=None):
        """Executes the pipeline step."""
        pass

class DataLoader(PipelineStep):
    def run(self, inputs=None):
        print("Running DataLoader...")
        # logic to download file
        file_path = self.config["input_path"]
        data = pd.read_csv(file_path)
        return data

class DataTransformer(PipelineStep):
    def run(self, inputs=None):
        input_data = inputs[0]
        print("Running DataTransformer...")
        # Transformations logic
        data = input_data.dropna()
        data = data[self.config['features'] + [self.config['target']]]
        return data

class DataSplitter(PipelineStep):
    def run(self, inputs=None):
        input_data = inputs[0]
        print("Running DataSplitter...")
        X = input_data[self.config['features']]
        y = input_data[self.config['target']]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config['test_size'], random_state=42
        )
        return {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}

class ModelTrainer(PipelineStep):
    def run(self, inputs=None):
        input_data = inputs[0]
        print("Running ModelTrainer...")
        model = RandomForestClassifier(**self.config['model_params'])
        model.fit(input_data['X_train'], input_data['y_train'])
        # Pass along split data and the trained model
        output_data = input_data
        output_data['model'] = model
        return output_data

class ModelEvaluator(PipelineStep):
    def run(self, inputs=None):
        input_data = inputs[0]
        print("Running ModelEvaluator...")
        model = input_data['model']
        score = model.score(input_data['X_test'], input_data['y_test'])
        print(f"Model Accuracy: {score:.4f}")
        # You might save metrics here or pass them along
        input_data['metrics'] = {'accuracy': score}
        # Often, only the model is needed for saving
        return input_data['model']

class ModelSaver(PipelineStep):
    def run(self, inputs=None):
        input_data = inputs[0]
        print("Running ModelSaver...")
        model = input_data  # Expecting the model object directly now
        # upload_file(model_path, self.config['bucket'], self.config['output_key'])
        # For example, saving locally:
        model_path = self.config['output_path']
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")
        return model_path  # Return path or confirmation

# Other classes for Plotting, Creating Files etc. similarly