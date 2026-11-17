import base64
import json
from datetime import datetime, timezone
import requests

TRACKER_PATH = "data/prediction_tracker.json"
DEFAULT_REPOSITORY = "retrovibexx/Tennis-edge-2.0"


def _headers(token):
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def load_tracker(token, repository=DEFAULT_REPOSITORY):
    if not token:
        raise RuntimeError("Add GITHUB_TOKEN to Streamlit Secrets.")
    url = f"https://api.github.com/repos/{repository}/contents/{TRACKER_PATH}"
    response = requests.get(url, headers=_headers(token), timeout=20)
    if response.status_code == 404:
        return {"predictions": []}, None
    response.raise_for_status()
    payload = response.json()
    content = base64.b64decode(payload["content"]).decode("utf-8")
    return json.loads(content), payload["sha"]


def save_tracker(token, tracker, repository=DEFAULT_REPOSITORY):
    current, sha = load_tracker(token, repository)
    url = f"https://api.github.com/repos/{repository}/contents/{TRACKER_PATH}"
    body = {
        "message": "Update Tennis Edge prediction accuracy tracker",
        "content": base64.b64encode(json.dumps(tracker, indent=2).encode("utf-8")).decode("ascii"),
    }
    if sha:
        body["sha"] = sha
    response = requests.put(url, headers=_headers(token), json=body, timeout=20)
    response.raise_for_status()
    return response.json()


def append_prediction(token, record, repository=DEFAULT_REPOSITORY):
    tracker, _ = load_tracker(token, repository)
    tracker.setdefault("predictions", []).append(record)
    return save_tracker(token, tracker, repository)


def grade_prediction(token, prediction_id, winner, repository=DEFAULT_REPOSITORY):
    tracker, _ = load_tracker(token, repository)
    for item in tracker.get("predictions", []):
        if item.get("id") == prediction_id:
            if item.get("actual_winner"):
                raise RuntimeError("This prediction already has a result.")
            item["actual_winner"] = winner
            item["resolved_at"] = datetime.now(timezone.utc).isoformat()
            item["correct"] = item.get("predicted_winner") == winner
            return save_tracker(token, tracker, repository)
    raise RuntimeError("Prediction record not found.")
