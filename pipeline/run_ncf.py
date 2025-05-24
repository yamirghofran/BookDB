import shutil
import subprocess
import os
import datetime
from utils import send_discord_webhook

def send_notification(title: str, description: str, color: int = 0x00FF00, fields: list = None, error: bool = False):
    """Send a Discord notification with consistent formatting."""
    try:
        embed = {
            "title": f"🎯 {title}" if not error else f"❌ {title}",
            "description": description,
            "color": color if not error else 0xFF0000,  # Red for errors
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "footer": {"text": "Pipeline Step: NCF Training"}
        }
        
        if fields:
            embed["fields"] = fields
            
        send_discord_webhook(
            content=None,
            embed=embed,
            username="BookDB Pipeline"
        )
    except Exception as e:
        print(f"Failed to send Discord notification: {e}")

# Send pipeline start notification
send_notification(
    "NCF Training Pipeline Started",
    "Beginning Neural Collaborative Filtering model training and export",
    color=0x0099FF,  # Blue for start
    fields=[
        {"name": "Training Script", "value": "`neural-collaborative-filtering/src/train.py`", "inline": True},
        {"name": "Export Script", "value": "`neural-collaborative-filtering/src/export_gmf_embeddings.py`", "inline": True},
        {"name": "Results Directory", "value": "`results/ncf`", "inline": True},
        {"name": "Model Type", "value": "Neural Collaborative Filtering", "inline": False}
    ]
)

try:
    # Create results directory if it doesn't exist
    os.makedirs("results/ncf", exist_ok=True)

    train_script = "neural-collaborative-filtering/src/train.py"
    
    # Send training start notification
    send_notification(
        "NCF Model Training Started",
        "Executing Neural Collaborative Filtering model training",
        color=0xFFA500,  # Orange for progress
        fields=[
            {"name": "Script", "value": f"`{train_script}`", "inline": True},
            {"name": "Status", "value": "🔄 Training in progress...", "inline": True}
        ]
    )

    # Run train.py
    result = subprocess.run(["python", train_script], check=True, capture_output=True, text=True)
    
    # Send training completion notification
    send_notification(
        "NCF Model Training Complete",
        "Successfully completed Neural Collaborative Filtering model training",
        fields=[
            {"name": "Training Script", "value": f"`{train_script}`", "inline": True},
            {"name": "Status", "value": "✅ Training completed", "inline": True},
            {"name": "Next Step", "value": "Exporting embeddings", "inline": True}
        ]
    )

    export_script = "neural-collaborative-filtering/src/export_gmf_embeddings.py"
    
    # Send export start notification
    send_notification(
        "NCF Embedding Export Started",
        "Exporting trained NCF model embeddings",
        color=0xFFA500,  # Orange for progress
        fields=[
            {"name": "Script", "value": f"`{export_script}`", "inline": True},
            {"name": "Status", "value": "🔄 Exporting embeddings...", "inline": True}
        ]
    )
    
    subprocess.run(["python", export_script], check=True, capture_output=True, text=True)
    
    # Send export completion notification
    send_notification(
        "NCF Embedding Export Complete",
        "Successfully exported NCF model embeddings",
        fields=[
            {"name": "Export Script", "value": f"`{export_script}`", "inline": True},
            {"name": "Status", "value": "✅ Export completed", "inline": True},
            {"name": "Output Directory", "value": "`results/ncf`", "inline": True}
        ]
    )

    print("NCF pipeline completed. Results saved to results/ncf directory.")
    
    # Send final success notification
    send_notification(
        "NCF Pipeline Complete! 🎉",
        "Successfully completed entire Neural Collaborative Filtering pipeline",
        color=0x00FF00,  # Green for success
        fields=[
            {"name": "Model Training", "value": "✅ Complete", "inline": True},
            {"name": "Embedding Export", "value": "✅ Complete", "inline": True},
            {"name": "Results Location", "value": "`results/ncf`", "inline": True},
            {"name": "Model Type", "value": "Neural Collaborative Filtering", "inline": True},
            {"name": "Status", "value": "🎯 Ready for recommendation serving", "inline": False}
        ]
    )

except subprocess.CalledProcessError as e:
    error_msg = f"NCF subprocess failed: {e.cmd} returned {e.returncode}"
    print(f"Error: {error_msg}")
    if e.stdout:
        print(f"STDOUT: {e.stdout}")
    if e.stderr:
        print(f"STDERR: {e.stderr}")
    
    send_notification(
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
    print(f"Error: {error_msg}")
    
    send_notification(
        "NCF Pipeline Failed",
        error_msg,
        error=True
    )
    raise