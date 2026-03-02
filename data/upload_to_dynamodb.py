"""
Upload doctor_schedules.json to AWS DynamoDB table.

Table:  DoctorSchedule (us-west-2)
PK:     department  (String)
SK:     doctor#day#time_slot  (String)

Usage:
    python data/upload_to_dynamodb.py
"""

import json
import os
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

# Load .env from project root (data/ → medi-agent/ → muti-agent/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-west-2")
TABLE_NAME = os.getenv("DYNAMODB_TABLE_NAME", "DoctorSchedule")

DATA_FILE = Path(__file__).resolve().parent / "processed" / "doctor_schedules.json"


def get_dynamodb_resource():
    return boto3.resource("dynamodb", region_name=AWS_REGION)


def ensure_table_exists(dynamodb):
    """Delete old table if schema mismatches, then create with correct schema."""
    client = dynamodb.meta.client

    try:
        desc = client.describe_table(TableName=TABLE_NAME)
        existing_keys = {k["AttributeName"] for k in desc["Table"]["KeySchema"]}
        expected_keys = {"department", "sk"}

        if existing_keys == expected_keys:
            print(f"Table '{TABLE_NAME}' already exists with correct schema.")
            return dynamodb.Table(TABLE_NAME)

        # Schema mismatch — delete and recreate
        print(f"Table '{TABLE_NAME}' has wrong schema {existing_keys}, expected {expected_keys}.")
        confirm = input(f"WARNING: This will DELETE all data in '{TABLE_NAME}'. Type table name to confirm: ")
        if confirm.strip() != TABLE_NAME:
            print("Aborted.")
            sys.exit(0)
        print(f"Deleting old table...")
        client.delete_table(TableName=TABLE_NAME)
        waiter = client.get_waiter("table_not_exists")
        waiter.wait(TableName=TABLE_NAME)
        print(f"Old table deleted.")

    except ClientError as e:
        if e.response["Error"]["Code"] != "ResourceNotFoundException":
            raise

    print(f"Creating table '{TABLE_NAME}'...")
    table = dynamodb.create_table(
        TableName=TABLE_NAME,
        KeySchema=[
            {"AttributeName": "department", "KeyType": "HASH"},   # Partition Key
            {"AttributeName": "sk", "KeyType": "RANGE"},          # Sort Key
        ],
        AttributeDefinitions=[
            {"AttributeName": "department", "AttributeType": "S"},
            {"AttributeName": "sk", "AttributeType": "S"},
        ],
        BillingMode="PAY_PER_REQUEST",
    )
    table.wait_until_exists()
    print(f"Table '{TABLE_NAME}' created successfully.")
    return table


def build_sort_key(record: dict) -> str:
    return f"{record['doctor']}#{record['day']}#{record['time_slot']}"


def upload_schedules():
    if not DATA_FILE.exists():
        print(f"ERROR: Data file not found: {DATA_FILE}")
        sys.exit(1)

    with open(DATA_FILE) as f:
        records = json.load(f)

    print(f"Loaded {len(records)} records from {DATA_FILE.name}")
    print(f"Target: {TABLE_NAME} ({AWS_REGION})")

    dynamodb = get_dynamodb_resource()
    table = ensure_table_exists(dynamodb)

    uploaded = 0
    with table.batch_writer() as batch:
        for record in records:
            item = {
                "department": record["department"],
                "sk": build_sort_key(record),
                "doctor": record["doctor"],
                "day": record["day"],
                "time_slot": record["time_slot"],
                "available": record["available"],
            }
            batch.put_item(Item=item)
            uploaded += 1

            if uploaded % 100 == 0:
                print(f"  ... {uploaded}/{len(records)} uploaded")

    print(f"Done! {uploaded} records uploaded to {TABLE_NAME}.")


if __name__ == "__main__":
    upload_schedules()
