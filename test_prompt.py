import requests
import time

prompt = "Imagine a fantasy video game scene in a mythical landscape with floating islands, cascading waterfalls, and ancient ruins covered in vines. Add an adventurer holding an ancient sword, standing on a jagged cliff as the sun sets."

print(f"Testing prompt: {prompt}")

start = time.time()
response = requests.post("http://localhost:8000/generate", json={"prompt": prompt})
end = time.time()

if response.status_code == 200:
    data = response.json()
    print(f"Success: {data['success']}")
    print(f"Run ID: {data['run_id']}")
    print(f"Time Taken: {end - start:.2f} seconds")
else:
    print(f"Failed with status code: {response.status_code}")
    print(response.text)
