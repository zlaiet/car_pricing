import requests
import json

# url = "http://127.0.0.1:2200/prediction"
url = "https://car-price-predictor-n.herokuapp.com/"
payload = json.dumps(
    {
        "kilometrage": 63000.0,
        "carrosserie": "compacte",
        "energie": "essence",
        "puissance": 6,
        "boite": "automatique",
        "transmission": "traction",
        "couleur": "noir",
        "model": "GOLF 7 ",
        "age": 2019,
        "brand": "VOLKSWAGEN",
    }
)
headers = {"Content-Type": "application/json"}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)
