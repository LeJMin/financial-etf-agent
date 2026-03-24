import csv
import json
import os
from datetime import datetime

CHAT_LOG_FILE = "logs/chat_log.csv"
ACCOUNT_LOG_FILE = "logs/account_log.jsonl"

os.makedirs("logs", exist_ok=True)


def save_log(question, route, response_time, success=True, user_name="unknown"):
    file_exists = os.path.isfile(CHAT_LOG_FILE)

    with open(CHAT_LOG_FILE, mode="a", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "user_name",
                "question",
                "route",
                "response_time",
                "success"
            ])

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            user_name,
            question,
            route,
            response_time,
            success
        ])


def save_account_log(user_name, account):
    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user_name": user_name,
        "account": account
    }

    with open(ACCOUNT_LOG_FILE, mode="a", encoding="utf-8-sig") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")