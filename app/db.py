import json
import os
from pathlib import Path

import boto3

AWS_REGION = os.getenv("AWS_REGION", "us-west-2")
DDB_TABLE = os.getenv("DDB_TABLE", "DoctorSchedule")

_LOCAL_SCHEDULES = Path(__file__).resolve().parents[1] / "data" / "processed" / "doctor_schedules.json"

dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
table = dynamodb.Table(DDB_TABLE)


def get_doctors_by_department(department: str):
    try:
        with open(_LOCAL_SCHEDULES) as f:
            schedules = json.load(f)
        return [e for e in schedules if e.get("department") == department]
    except Exception:
        pass

    try:
        resp = table.scan(
            FilterExpression="department = :dept",
            ExpressionAttributeValues={":dept": department},
        )
        return resp.get("Items", [])
    except Exception:
        return []
