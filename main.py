from pipeline import DataLoader, DataTransformer, ModelTrainer, ModelEvaluator, PipelineStep, DataSplitter, ModelSaver
from utils import load_config

config = load_config("config.yaml")

steps = [
    DataLoader(config['data_loader']),
    DataTransformer(config['data_transformer']),
    DataSplitter(config['data_splitter']),
    ModelTrainer(config['model_trainer']),
    ModelEvaluator(config['model_evaluator']),
    ModelSaver(config['model_evaluator']),
]

pipeline = PipelineStep(config)
final_output = pipeline.run()