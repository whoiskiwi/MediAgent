import os
import boto3

AWS_REGION = os.getenv("AWS_REGION", "us-west-2")
DDB_TABLE = os.getenv("DDB_TABLE", "DoctorSchedule")

dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
table = dynamodb.Table(DDB_TABLE)

def get_doctors_by_department(department: str):
    resp = table.scan(
        FilterExpression="department = :dept",
        ExpressionAttributeValues={":dept": department},
    )
    return resp.get("Items", [])