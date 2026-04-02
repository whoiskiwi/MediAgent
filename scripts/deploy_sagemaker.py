"""
Step 2 — Package merged models → upload to S3 → deploy SageMaker endpoints.
Uses boto3 directly (no sagemaker SDK dependency).

Creates two endpoints:
  medi-agent-classifier   (Agent 1)
  medi-agent-generator    (Agent 3)

Usage:
    python scripts/deploy_sagemaker.py

Required .env variables:
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION
    S3_BUCKET              e.g. medi-agent-models-156041402579
    SAGEMAKER_ROLE_ARN     e.g. arn:aws:iam::123456789:role/SageMakerRole
"""
import os
import sys
import tarfile
import shutil
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

import boto3

AWS_REGION    = os.getenv("AWS_REGION", "us-west-2")
S3_BUCKET     = os.getenv("S3_BUCKET")
ROLE_ARN      = os.getenv("SAGEMAKER_ROLE_ARN")
INSTANCE_TYPE = os.getenv("SAGEMAKER_INSTANCE", "ml.g4dn.xlarge")

# Docker image: GPU or CPU depending on instance type
_GPU_IMAGE = (
    "763104351884.dkr.ecr.us-west-2.amazonaws.com/"
    "huggingface-pytorch-inference:2.6.0-transformers4.49.0-gpu-py312-cu124-ubuntu22.04"
)
_CPU_IMAGE = (
    "763104351884.dkr.ecr.us-west-2.amazonaws.com/"
    "huggingface-pytorch-inference:2.6.0-transformers4.49.0-cpu-py312-ubuntu22.04"
)
HF_IMAGE = _CPU_IMAGE if INSTANCE_TYPE.startswith("ml.m") else _GPU_IMAGE

INFERENCE_SCRIPT = ROOT / "sagemaker_code" / "inference.py"

JOBS = [
    {
        "name":      "classifier",
        "model_dir": ROOT / "data/processed/merged_classifier",
        "s3_prefix": "medi-agent/classifier",
        "endpoint":  "medi-agent-classifier",
    },
    {
        "name":      "generator",
        "model_dir": ROOT / "data/processed/merged_generator",
        "s3_prefix": "medi-agent/generator",
        "endpoint":  "medi-agent-generator",
    },
]

s3 = boto3.client("s3", region_name=AWS_REGION)
sm = boto3.client("sagemaker", region_name=AWS_REGION)


def _make_tarball(model_dir: Path, job_name: str) -> Path:
    tmp = ROOT / "tmp_tar" / job_name
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    # Copy model files into tmp/
    shutil.copytree(str(model_dir), str(tmp), dirs_exist_ok=True)
    # SageMaker looks for code/inference.py inside the tarball
    code_dir = tmp / "code"
    code_dir.mkdir(exist_ok=True)
    shutil.copy(INFERENCE_SCRIPT, code_dir / "inference.py")
    req_file = ROOT / "sagemaker_code" / "requirements.txt"
    if req_file.exists():
        shutil.copy(req_file, code_dir / "requirements.txt")

    tarball = ROOT / "tmp_tar" / f"{job_name}.tar.gz"
    with tarfile.open(tarball, "w:gz") as tar:
        for item in tmp.iterdir():
            tar.add(item, arcname=item.name)

    size_gb = tarball.stat().st_size / 1e9
    print(f"[deploy] Created {tarball.name} ({size_gb:.1f} GB)")
    return tarball


def _upload(tarball: Path, s3_prefix: str) -> str:
    s3_key = f"{s3_prefix}/model.tar.gz"
    print(f"[deploy] Uploading to s3://{S3_BUCKET}/{s3_key} ...")
    s3.upload_file(str(tarball), S3_BUCKET, s3_key)
    uri = f"s3://{S3_BUCKET}/{s3_key}"
    print(f"[deploy] Upload done: {uri}")
    return uri


def _deploy(s3_uri: str, endpoint_name: str):
    model_name   = f"{endpoint_name}-model"
    config_name  = f"{endpoint_name}-config"

    # Force delete old model and config to ensure new inference.py is used
    try:
        sm.delete_model(ModelName=model_name)
    except Exception:
        pass
    try:
        sm.delete_endpoint_config(EndpointConfigName=config_name)
    except Exception:
        pass

    # Create model
    print(f"[deploy] Creating SageMaker model: {model_name}")
    sm.create_model(
        ModelName=model_name,
        PrimaryContainer={
            "Image": HF_IMAGE,
            "ModelDataUrl": s3_uri,
            "Environment": {"SAGEMAKER_PROGRAM": "inference.py"},
        },
        ExecutionRoleArn=ROLE_ARN,
    )

    # Create endpoint config
    print(f"[deploy] Creating endpoint config: {config_name}")
    sm.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[{
            "VariantName":    "AllTraffic",
            "ModelName":      model_name,
            "InstanceType":   INSTANCE_TYPE,
            "InitialInstanceCount": 1,
        }],
    )

    # Create or update endpoint
    print(f"[deploy] Deploying endpoint: {endpoint_name} (this takes ~10 min) ...")
    endpoint_exists = False
    try:
        sm.describe_endpoint(EndpointName=endpoint_name)
        endpoint_exists = True
    except Exception:
        pass

    if endpoint_exists:
        print(f"  Endpoint exists — updating ...")
        sm.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name,
        )
    else:
        sm.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name,
        )

    # Wait for InService
    print(f"[deploy] Waiting for {endpoint_name} to be InService ...")
    waiter = sm.get_waiter("endpoint_in_service")
    waiter.wait(EndpointName=endpoint_name, WaiterConfig={"Delay": 30, "MaxAttempts": 40})
    print(f"[deploy] ✓ {endpoint_name} is InService")


def deploy(job: dict):
    print(f"\n[deploy] === {job['name']} ===")
    model_dir = Path(job["model_dir"])
    if not model_dir.exists():
        print(f"ERROR: merged model not found at {model_dir}")
        print("Run: python scripts/merge_adapter.py first")
        sys.exit(1)

    tarball = _make_tarball(model_dir, job["name"])
    s3_uri  = _upload(tarball, job["s3_prefix"])
    _deploy(s3_uri, job["endpoint"])


if __name__ == "__main__":
    for var, val in [("S3_BUCKET", S3_BUCKET), ("SAGEMAKER_ROLE_ARN", ROLE_ARN)]:
        if not val:
            print(f"ERROR: {var} not set in .env")
            sys.exit(1)

    for job in JOBS:
        deploy(job)

    shutil.rmtree(ROOT / "tmp_tar", ignore_errors=True)
    print("\n[deploy] All endpoints live.")
    print("  AGENT1_ENDPOINT_NAME=medi-agent-classifier")
    print("  AGENT3_ENDPOINT_NAME=medi-agent-generator")
