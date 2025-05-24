import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Type, Optional
import os

class PipelineStep(ABC):
    """Base class for all pipeline steps."""
    
    def __init__(self, name: str):
        """
        Initialize a pipeline step.
        
        Args:
            name: Name identifier for this step (matches config section)
        """
        self.name = name
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = {}
        self.input_data = {}
        self.output_data = {}
        
    @abstractmethod
    def run(self) -> Dict[str, Any]:
        """
        Execute the functionality of the pipeline step.
        
        Returns:
            Dictionary containing output data from this step
        """
        pass
    
    def configure(self, config: Any) -> None:
        """
        Configure this step with values from config object.
        
        Args:
            config: Configuration object for this step (can be dataclass or dict)
        """
        if hasattr(config, '__dict__'):
            # Convert dataclass to dict
            self.config = config.__dict__
        else:
            # Assume it's already a dict
            self.config = config
        self._validate_config()
        
    def _validate_config(self) -> None:
        """
        Validate the configuration.
        Subclasses should override this to implement specific validation.
        """
        pass
        
    def set_input(self, input_data: Dict[str, Any]) -> None:
        """Set input data from previous step."""
        self.input_data = input_data
        
    def get_output(self) -> Dict[str, Any]:
        """Get output data produced by this step."""
        return self.output_data


class Pipeline:
    """Orchestrates a sequence of pipeline steps with Python configuration."""
    
    def __init__(self, config):
        """
        Initialize pipeline with configuration object.
        
        Args:
            config: PipelineConfig object or path to Python config module
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        
        if isinstance(config, str):
            # If config is a string, treat it as a module path and import
            self.config = self._load_config_from_module(config)
        else:
            # Assume it's already a config object
            self.config = config
            
        self.steps = []
        self._setup_logging()
        
    def _load_config_from_module(self, module_path: str):
        """Load configuration from a Python module."""
        try:
            import importlib
            module = importlib.import_module(module_path)
            if hasattr(module, 'PipelineConfig'):
                config = module.PipelineConfig()
                self.logger.info(f"Loaded configuration from {module_path}")
                return config
            else:
                raise AttributeError(f"Module {module_path} does not have PipelineConfig class")
        except Exception as e:
            self.logger.error(f"Failed to load config from {module_path}: {str(e)}")
            raise
    
    def _setup_logging(self):
        """Setup logging based on configuration."""
        log_level = getattr(self.config.global_config, 'log_level', 'INFO')
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
            
    def add_step(self, step_class: Type[PipelineStep], step_name: str) -> None:
        """
        Add a step to the pipeline.
        
        Args:
            step_class: PipelineStep class to instantiate
            step_name: Name of the step (must match a config attribute)
        """
        if not hasattr(self.config, step_name):
            self.logger.warning(f"No configuration found for step '{step_name}'")
            step_config = {}
        else:
            step_config = getattr(self.config, step_name)
            
        step = step_class(step_name)
        step.configure(step_config)
        self.steps.append(step)
        self.logger.info(f"Added step: {step_name}")
        
    def run(self, initial_input: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run the pipeline from start to finish.
        
        Args:
            initial_input: Optional initial input for first step
            
        Returns:
            Output from final step
        """
        current_input = initial_input or {}
        final_output = {}
        
        for i, step in enumerate(self.steps):
            self.logger.info(f"Running step {i+1}/{len(self.steps)}: {step.name}")
            
            # Set input from previous step
            step.set_input(current_input)
            
            # Execute the step
            try:
                step_output = step.run()
                current_input = step_output
                final_output = step_output
                self.logger.info(f"Step completed: {step.name}")
            except Exception as e:
                self.logger.error(f"Error in step {step.name}: {str(e)}")
                raise
                
        return final_output