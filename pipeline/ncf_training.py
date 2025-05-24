import subprocess
import os
import datetime
from typing import Dict, Any
from .core import PipelineStep
from utils import send_discord_webhook

class NCFTrainingStep(PipelineStep):
    def __init__(self, name: str):
        super().__init__(name)
        # Default configuration
        self.train_script = "neural-collaborative-filtering/src/train.py"
        self.export_script = "neural-collaborative-filtering/src/export_gmf_embeddings.py"
        self.results_dir = "results/ncf"
    
    def configure(self, config: Dict[str, Any]) -> None:
        super().configure(config)
        self.train_script = self.config.get("train_script", self.train_script)
        self.export_script = self.config.get("export_script", self.export_script)
        self.results_dir = self.config.get("results_dir", self.results_dir)
        # Ensure results directory exists
        os.makedirs(self.results_dir, exist_ok=True)
    
    def _send_notification(self, title: str, description: str, color: int = 0x00FF00, fields: list = None, error: bool = False):
        """Send a Discord notification with consistent formatting."""
        try:
            embed = {
                "title": f"🎯 {title}" if not error else f"❌ {title}",
                "description": description,
                "color": color if not error else 0xFF0000,  # Red for errors
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "footer": {"text": f"Pipeline Step: {self.name}"}
            }
            
            if fields:
                embed["fields"] = fields
                
            send_discord_webhook(
                content=None,
                embed=embed,
                username="BookDB Pipeline"
            )
        except Exception as e:
            self.logger.warning(f"Failed to send Discord notification: {e}")
    
    def _run_training(self):
        """Run the NCF training script."""
        self._send_notification(
            "NCF Model Training Started",
            "Executing Neural Collaborative Filtering model training",
            color=0xFFA500,  # Orange for progress
            fields=[
                {"name": "Script", "value": f"`{self.train_script}`", "inline": True},
                {"name": "Status", "value": "🔄 Training in progress...", "inline": True}
            ]
        )

        result = subprocess.run(["python", self.train_script], check=True, capture_output=True, text=True)
        
        self._send_notification(
            "NCF Model Training Complete",
            "Successfully completed Neural Collaborative Filtering model training",
            fields=[
                {"name": "Training Script", "value": f"`{self.train_script}`", "inline": True},
                {"name": "Status", "value": "✅ Training completed", "inline": True},
                {"name": "Next Step", "value": "Exporting embeddings", "inline": True}
            ]
        )
        
        return result
    
    def _run_export(self):
        """Run the NCF embedding export script."""
        self._send_notification(
            "NCF Embedding Export Started",
            "Exporting trained NCF model embeddings",
            color=0xFFA500,  # Orange for progress
            fields=[
                {"name": "Script", "value": f"`{self.export_script}`", "inline": True},
                {"name": "Status", "value": "🔄 Exporting embeddings...", "inline": True}
            ]
        )
        
        result = subprocess.run(["python", self.export_script], check=True, capture_output=True, text=True)
        
        self._send_notification(
            "NCF Embedding Export Complete",
            "Successfully exported NCF model embeddings",
            fields=[
                {"name": "Export Script", "value": f"`{self.export_script}`", "inline": True},
                {"name": "Status", "value": "✅ Export completed", "inline": True},
                {"name": "Output Directory", "value": f"`{self.results_dir}`", "inline": True}
            ]
        )
        
        return result
    
    def run(self) -> Dict[str, Any]:
        self.logger.info(f"Starting NCF training step: {self.name}")
        
        # Send pipeline start notification
        self._send_notification(
            "NCF Training Pipeline Started",
            f"Beginning Neural Collaborative Filtering model training and export: **{self.name}**",
            color=0x0099FF,  # Blue for start
            fields=[
                {"name": "Training Script", "value": f"`{self.train_script}`", "inline": True},
                {"name": "Export Script", "value": f"`{self.export_script}`", "inline": True},
                {"name": "Results Directory", "value": f"`{self.results_dir}`", "inline": True},
                {"name": "Model Type", "value": "Neural Collaborative Filtering", "inline": False}
            ]
        )
        
        try:
            outputs = {}
            
            # Run training
            train_result = self._run_training()
            outputs["training_result"] = train_result
            
            # Run export
            export_result = self._run_export()
            outputs["export_result"] = export_result
            outputs["results_directory"] = self.results_dir
            
            self.logger.info(f"NCF training step {self.name} finished successfully.")
            
            # Send final success notification
            self._send_notification(
                "NCF Pipeline Complete! 🎉",
                f"Successfully completed entire Neural Collaborative Filtering pipeline: **{self.name}**",
                color=0x00FF00,  # Green for success
                fields=[
                    {"name": "Model Training", "value": "✅ Complete", "inline": True},
                    {"name": "Embedding Export", "value": "✅ Complete", "inline": True},
                    {"name": "Results Location", "value": f"`{self.results_dir}`", "inline": True},
                    {"name": "Model Type", "value": "Neural Collaborative Filtering", "inline": True},
                    {"name": "Status", "value": "🎯 Ready for recommendation serving", "inline": False}
                ]
            )
            
            self.output_data = outputs
            return outputs
            
        except subprocess.CalledProcessError as e:
            error_msg = f"NCF subprocess failed: {e.cmd} returned {e.returncode}"
            self.logger.error(error_msg)
            if e.stdout:
                self.logger.error(f"STDOUT: {e.stdout}")
            if e.stderr:
                self.logger.error(f"STDERR: {e.stderr}")
            
            self._send_notification(
                "NCF Pipeline Failed",
                error_msg,
                error=True,
                fields=[
                    {"name": "Failed Command", "value": f"`{' '.join(e.cmd)}`", "inline": True},
                    {"name": "Return Code", "value": f"{e.returncode}", "inline": True},
                    {"name": "Error Output", "value": f"```{e.stderr[:500] if e.stderr else 'No error output'}```", "inline": False}
                ]
            )
            raise

        except Exception as e:
            error_msg = f"NCF pipeline failed with unexpected error: {str(e)}"
            self.logger.error(error_msg)
            
            self._send_notification(
                "NCF Pipeline Failed",
                error_msg,
                error=True
            )
            raise 