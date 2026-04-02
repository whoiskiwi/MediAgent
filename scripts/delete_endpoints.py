"""
Delete SageMaker endpoints, endpoint configs, and models to stop billing.

Usage:
    python scripts/delete_endpoints.py
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

import boto3

AWS_REGION = os.getenv("AWS_REGION", "us-west-2")
sm = boto3.client("sagemaker", region_name=AWS_REGION)

ENDPOINTS = ["medi-agent-classifier", "medi-agent-generator"]


def delete_endpoint(name: str):
    model_name  = f"{name}-model"
    config_name = f"{name}-config"

    for resource, fn in [
        ("endpoint",        lambda: sm.delete_endpoint(EndpointName=name)),
        ("endpoint config", lambda: sm.delete_endpoint_config(EndpointConfigName=config_name)),
        ("model",           lambda: sm.delete_model(ModelName=model_name)),
    ]:
        try:
            fn()
            print(f"[delete] Deleted {resource}: {name}")
        except sm.exceptions.ClientError if hasattr(sm, 'exceptions') else Exception:
            print(f"[delete] {resource} not found (already deleted): {name}")
        except Exception:
            print(f"[delete] {resource} not found (already deleted): {name}")


if __name__ == "__main__":
    for ep in ENDPOINTS:
        delete_endpoint(ep)
    print("\n[delete] Done. SageMaker billing stopped.")
