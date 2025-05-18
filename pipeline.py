import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Type, Optional
import yaml
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
    
    def configure(self, config: Dict[str, Any]) -> None:
        """
        Configure this step with values from config file.
        
        Args:
            config: Configuration dictionary for this step
        """
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
    """Orchestrates a sequence of pipeline steps with automatic config loading."""
    
    def __init__(self, config_path: str):
        """
        Initialize pipeline with path to config file.
        
        Args:
            config_path: Path to YAML config file
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config_path = config_path
        self.config = self._load_config()
        self.steps = []
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
                self.logger.info(f"Loaded configuration from {self.config_path}")
                return config
        except Exception as e:
            self.logger.error(f"Failed to load config from {self.config_path}: {str(e)}")
            raise
            
    def add_step(self, step_class: Type[PipelineStep], step_name: str) -> None:
        """
        Add a step to the pipeline.
        
        Args:
            step_class: PipelineStep class to instantiate
            step_name: Name of the step (must match a section in config)
        """
        if step_name not in self.config:
            self.logger.warning(f"No configuration found for step '{step_name}'")
            
        step = step_class(step_name)
        step.configure(self.config.get(step_name, {}))
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
            
            # Process the step
            try:
                step_output = step.process()
                current_input = step_output
                final_output = step_output
                self.logger.info(f"Step completed: {step.name}")
            except Exception as e:
                self.logger.error(f"Error in step {step.name}: {str(e)}")
                raise
                
        return final_output