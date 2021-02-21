import requests

url = 'http://localhost:5000/results'
r = requests.post(url,json={"V4":1.38,"V10":0.09,"V11":-0.55,"V12":-0.62,"V14":-0.31,"V17":0.21})

print(r.json())