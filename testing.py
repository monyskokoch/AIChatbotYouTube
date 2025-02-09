import requests

url = "http://127.0.0.1:5000/chat"
data = {"message": "How do I get more attractive?"}
headers = {"Content-Type": "application/json"}

response = requests.post(url, json=data, headers=headers)
print(response.json())  # ✅ This should return the AI-generated response
