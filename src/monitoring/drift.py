import os
import yaml
import logging
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import mlflow
from src.database.connection import SessionLocal
from src.database.models import DriftMetric

logger = logging.getLogger(__name__)

# Try Evidently imports
HAS_EVIDENTLY = False
try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, TextOverviewPreset
    HAS_EVIDENTLY = True
except ImportError:
    pass

# Load configuration
config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "configs", "pipeline_config.yaml")
mlflow_uri = "http://localhost:5000"
experiment_name = "customer_voice_intelligence"

if os.path.exists(config_path):
    try:
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
            if cfg and 'models' in cfg and 'mlflow' in cfg['models']:
                mlflow_uri = cfg['models']['mlflow'].get('tracking_uri', mlflow_uri)
                experiment_name = cfg['models']['mlflow'].get('experiment_name', experiment_name)
    except Exception as e:
        logger.warning(f"Failed to load MLflow config: {e}")

class ModelMonitor:
    def __init__(self):
        # Configure MLflow
        try:
            mlflow.set_tracking_uri(mlflow_uri)
            mlflow.set_experiment(experiment_name)
        except Exception as e:
            logger.warning(f"Could not connect to MLflow server at {mlflow_uri}: {e}. Operating in local offline mode.")

    def log_inference_run(self, name: str, params: dict, metrics: dict):
        """Logs model inference parameters and performance metrics to MLflow."""
        try:
            with mlflow.start_run(run_name=name):
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)
                logger.info(f"Successfully logged run '{name}' to MLflow.")
        except Exception as e:
            logger.debug(f"Failed to log run to MLflow offline mode: {e}")

    def run_drift_analysis(self, reference_df: pd.DataFrame, current_df: pd.DataFrame) -> dict:
        """
        Runs text data drift analysis.
        Uses EvidentlyAI if available, otherwise falls back to custom statistical checks (KS test, distribution divergence).
        """
        if reference_df.empty or current_df.empty:
            return {"drift_detected": False, "status": "Empty datasets"}
            
        report_data = {}
        drift_detected = False
        
        # 1. Evidently Report
        if HAS_EVIDENTLY:
            try:
                report = Report(metrics=[DataDriftPreset()])
                report.run(reference_data=reference_df, current_data=current_df)
                report_dict = report.dict()
                
                # Check for overall drift
                dataset_drift = report_dict["metrics"][0]["result"]["dataset_drift"]
                drift_detected = dataset_drift
                report_data = report_dict
                logger.info("EvidentlyAI data drift check completed successfully.")
            except Exception as e:
                logger.warning(f"Evidently report computation failed: {e}. Reverting to statistical fallback...")
                
        # 2. Custom Statistical Drift Check (Fallback)
        if not report_data:
            logger.info("Running statistical text distribution checks (KS-test & Label divergence)...")
            
            # Check Text Length Drift (numerical drift check using Kolmogorov-Smirnov test)
            ref_len = reference_df["text"].str.len()
            cur_len = current_df["text"].str.len()
            ks_stat, p_val = ks_2samp(ref_len, cur_len)
            length_drift = p_val < 0.05 # P-value threshold
            
            # Check Sentiment Label Divergence (categorical drift check)
            ref_sent = reference_df["sentiment"].value_counts(normalize=True).to_dict()
            cur_sent = current_df["sentiment"].value_counts(normalize=True).to_dict()
            
            sent_diff = 0.0
            for label in set(ref_sent.keys()).union(cur_sent.keys()):
                diff = abs(ref_sent.get(label, 0.0) - cur_sent.get(label, 0.0))
                sent_diff += diff
                
            sent_drift = sent_diff > 0.15 # 15% shift threshold
            drift_detected = length_drift or sent_drift
            
            report_data = {
                "length_ks_statistic": float(ks_stat),
                "length_p_value": float(p_val),
                "length_drift_detected": bool(length_drift),
                "sentiment_distribution_divergence": float(sent_diff),
                "sentiment_drift_detected": bool(sent_drift),
                "overall_drift_detected": bool(drift_detected)
            }
            
        # Log drift metrics to SQL Database
        db = SessionLocal()
        try:
            import json
            metric_val = 1.0 if drift_detected else 0.0
            dm = DriftMetric(
                metric_name="dataset_drift",
                value=metric_val,
                report_json=json.dumps(report_data)
            )
            db.add(dm)
            db.commit()
            logger.info(f"Log data drift metric into db: {metric_val}")
        except Exception as e:
            logger.warning(f"Failed to log drift metrics to db: {e}")
        finally:
            db.close()
            
        return {
            "drift_detected": drift_detected,
            "metrics": report_data
        }

    def trigger_automated_retraining(self, new_data_path: str):
        """Mock trigger for automated model retraining pipelines when drift is critical."""
        logger.warning(f"CRITICAL DRIFT WARNING: Automatically triggering DVC updates and model retraining workflow for dataset: {new_data_path}")
        # In a real workflow, this would call a subprocess to run DVC commands
        # or execute a GitHub action trigger API call.
