import requests

data = {
  "amount_log": 3.2,
  "hour": 15,
  "Amount": 120
}

resp = requests.post("http://127.0.0.1:5000/predict", json=data)
print(resp.json())
