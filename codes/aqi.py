import requests
import datetime
import time
import json

API_KEYS = ["86171d9f-0135-4c78-8c4b-e46b9613923c", "3541392e-7a81-4715-9731-89c00a886ef9"]
CITY = "Abuja"
STATE = "FCT"
COUNTRY = "Nigeria"

def get_air_quality_data(api_key):
    api_url = f"http://api.airvisual.com/v2/city?city={CITY}&state={STATE}&country={COUNTRY}&key={api_key}"
    response = requests.get(api_url)
    return response.json()

if __name__ == "__main__":
    DATA_FILE = "air_quality_data.json"
    interval = 60*30  # .5 hour in seconds
    api_key_index = 0

    while True:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        current_api_key = API_KEYS[api_key_index]
        
        #print(f"Collecting data at {current_time} using API key: {current_api_key}")

        air_quality_data = get_air_quality_data(current_api_key)
        #print("Data collected:\n", air_quality_data)

        with open(DATA_FILE, "a") as file:
            if file.tell() != 0:
                file.write(",\n")  # Separate entries with comma and newline
            json.dump(air_quality_data, file)
        
        #print("Data appended to JSON file.\n")

        api_key_index = (api_key_index + 1) % len(API_KEYS)  # Cycle through keys
        time.sleep(interval)