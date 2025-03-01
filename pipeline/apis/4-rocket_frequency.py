#!/usr/bin/env python3
"""Pipeline Api"""
import requests


if __name__ == '__main__':
    """pipeline api"""
    url = "https://api.spacexdata.com/v4/launches"
    r = requests.get(url)
    rocket_dict = {}

    for launch in r.json():
        rocket_id = launch["rocket"]
        rocket_dict[rocket_id] = rocket_dict.get(rocket_id, 0) + 1

    # Sort by count (descending) and name (ascending) when counts are equal
    sorted_rockets = sorted(
        rocket_dict.items(),
        key=lambda kv: (-kv[1], kv[0])  # Negative count for descending, ID for tie-breaking
    )

    for key, value in sorted_rockets:
        rurl = f"https://api.spacexdata.com/v4/rockets/{key}"
        req = requests.get(rurl)

        print(req.json()["name"] + ": " + str(value))
