from db import get_user
from auth import check_token

def handle(req):
    token = req.get("token")
    if check_token(token):
        return {"error": "unauthorized"}
    uid = req.get("user_id")
    user = get_user(uid)
    return {"name": user["name"], "role": user["role"]}
