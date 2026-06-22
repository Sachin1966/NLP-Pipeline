import os
import logging
import datetime
import json

# Create log directories
log_dir = "data/logs"
os.makedirs(log_dir, exist_ok=True)
alert_file = os.path.join(log_dir, "alerts.json")

# Configure logger
log_file = os.path.join(log_dir, "platform.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("voice_intelligence")

def trigger_alert(module: str, level: str, message: str):
    """
    Writes an alert log event to the alerts.json file.
    Alert levels: INFO, WARNING, ERROR, CRITICAL
    """
    alert_event = {
        "timestamp": datetime.datetime.now().isoformat(),
        "module": module,
        "level": level,
        "message": message
    }
    
    logger.warning(f"ALERT [{level}] in {module}: {message}")
    
    # Read existing alerts
    alerts = []
    if os.path.exists(alert_file):
        try:
            with open(alert_file, 'r') as f:
                alerts = json.load(f)
        except Exception:
            pass
            
    alerts.append(alert_event)
    
    # Keep last 50 alerts only to conserve memory
    alerts = alerts[-50:]
    
    try:
        with open(alert_file, 'w') as f:
            json.dump(alerts, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save system alert: {e}")

def get_alerts() -> list:
    """Retrieves list of active logs system alerts."""
    if os.path.exists(alert_file):
        try:
            with open(alert_file, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return []
