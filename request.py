import requests

url = 'http://localhost:5000/results'
r = requests.post(url,json={"Sessions":304, "Avg. Session Duration":925.82, "Bounce Rate":0.18})

print(r.json())