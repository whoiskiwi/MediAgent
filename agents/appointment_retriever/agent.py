"""Agent 2: Appointment Retriever

Queries the DynamoDB DoctorSchedule table to find available
doctor appointments for a given department.
"""

import os

import boto3
from boto3.dynamodb.conditions import Key
from dotenv import load_dotenv

from schemas import AppointmentOutput, AppointmentQuery

load_dotenv()

TABLE_NAME = os.getenv("DYNAMODB_TABLE_NAME", "DoctorSchedule")
AWS_REGION = os.getenv("AWS_REGION", "us-west-2")
DAY_ORDER = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4}


class AppointmentRetrieverAgent:
    def __init__(self):
        self.dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
        self.table = self.dynamodb.Table(TABLE_NAME)

    def retrieve(self, query: AppointmentQuery) -> AppointmentOutput:
        response = self.table.query(
            KeyConditionExpression=Key("department").eq(query.department),
            FilterExpression="available = :avail",
            ExpressionAttributeValues={":avail": True},
        )

        items = response.get("Items", [])
        if not items:
            return AppointmentOutput(
                doctor="No available doctor",
                time_slot="No available slot",
            )

        # Pick the first available slot (earliest day + time)
        items.sort(key=lambda x: (DAY_ORDER.get(x["day"], 99), x["time_slot"]))

        chosen = items[0]
        return AppointmentOutput(
            doctor=chosen["doctor"],
            time_slot=f"{chosen['day']} at {chosen['time_slot']}",
        )
