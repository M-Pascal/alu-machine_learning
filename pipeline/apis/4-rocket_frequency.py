#!/usr/bin/env python3
"""Pipeline Api"""
import requests


if __name__ == '__main__':
    """pipeline api"""
    url = "https://api.spacexdata.com/v5/launches"
    r = requests.get(url)
    rocket_dict = {}

    for launch in r.json():
        rocket_id = launch["rocket"]
        if rocket_id in rocket_dict:
            rocket_dict[rocket_id] += 1
        else:
            rocket_dict[rocket_id] = 1

    # Fetch rocket names and print results
    for key, value in sorted(rocket_dict.items(), key=lambda kv: kv[1], reverse=True):
        rurl = "https://api.spacexdata.com/v5/rockets/" + key
        req = requests.get(rurl)
        rocket_name = req.json().get("name", "Not Found")
        print(f"{rocket_name}: {value}")

    # Additional output
    print("Today is Saturday, March 1, 2025 and here are the results:")
