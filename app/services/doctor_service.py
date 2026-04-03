from app.db import table

def get_doctors_by_department(department: str):
    resp = table.scan(
        FilterExpression="department = :dept",
        ExpressionAttributeValues={":dept": department},
    )
    return resp.get("Items", [])