import requests
import json

url = "http://localhost:8000/api/v1/retrieve"
payload = {
    "query": "biển báo",
    "mode": "agent",
    "top_k": 10
}
try:
    response = requests.post(url, json=payload)
    print("Status Code:", response.status_code)
    with open("api_response.json", "w", encoding="utf-8") as f:
        f.write(response.text)
    data = response.json()
    print("Found items:", len(data))
except Exception as e:
    print("Error:", e)
