import requests
import json
import time

def test_health():
    print("Testing API Health (/)...")
    try:
        response = requests.get("http://localhost:8000/")
        print("Status Code:", response.status_code)
        print("Response:", json.dumps(response.json(), indent=2))
        return True
    except Exception as e:
        print("API is offline or unreachable:", e)
        return False

def test_generate():
    print("\nTesting Image Generation (/generate)...")
    payload = {
        "prompt": "three red apples on a wooden table"
    }
    
    start_time = time.time()
    response = requests.post("http://localhost:8000/generate", json=payload)
    duration = time.time() - start_time
    
    print(f"Status Code: {response.status_code}")
    print(f"Time Taken: {duration:.2f} seconds")
    
    if response.status_code == 200:
        data = response.json()
        print("\n--- Verification Report ---")
        print(json.dumps(data["verification_report"], indent=2))
        print(f"\nSuccess: {data['success']}")
        print(f"Run ID: {data['run_id']}")
        if data.get("image_base64"):
            print("[Image successfully returned as base64]")
    else:
        print("Error:", response.text)

if __name__ == "__main__":
    if test_health():
        test_generate()
