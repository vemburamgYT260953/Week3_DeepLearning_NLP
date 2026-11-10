
import requests

BASE_URL = "http://127.0.0.1:8000"


# Test 1: Home endpoint
print("\nTEST 1: HOME")

response = requests.get(f"{BASE_URL}/")

print("Status:", response.status_code)
print("Response:", response.json())


# Test 2: Health endpoint
print("\nTEST 2: HEALTH")

response = requests.get(f"{BASE_URL}/health")

print("Status:", response.status_code)
print("Response:", response.json())


# Test 3: Positive feedback
print("\nTEST 3: POSITIVE FEEDBACK")

response = requests.post(
    f"{BASE_URL}/predict",
    json={"text": "Excellent customer service. I am very happy!"}
)

print("Status:", response.status_code)
print("Response:", response.json())


# Test 4: Negative feedback
print("\nTEST 4: NEGATIVE FEEDBACK")

response = requests.post(
    f"{BASE_URL}/predict",
    json={"text": "Very poor service. I am disappointed."}
)

print("Status:", response.status_code)
print("Response:", response.json())


# Test 5: Neutral feedback
print("\nTEST 5: NEUTRAL FEEDBACK")

response = requests.post(
    f"{BASE_URL}/predict",
    json={"text": "The package arrived yesterday."}
)

print("Status:", response.status_code)
print("Response:", response.json())


# Test 6: Empty text validation
print("\nTEST 6: EMPTY TEXT")

response = requests.post(
    f"{BASE_URL}/predict",
    json={"text": ""}
)

print("Status:", response.status_code)
print("Response:", response.json())
