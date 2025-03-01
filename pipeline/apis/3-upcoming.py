#!/usr/bin/env python3
"""
Uses the (unofficial) SpaceX API to print the number of launches per rocket as:
<rocket name>: <number of launches>
ordered by the number of launches in descending order or,
if rockets have the same amount of launches, in alphabetical order
"""

import requests

if __name__ == "__main__":
    url = 'https://api.spacexdata.com/v4/launches'
    launches = requests.get(url).json()
    
    rocket_counts = {}
    
    for launch in launches:
        rocket_id = launch.get('rocket')
        if rocket_id not in rocket_counts:
            rocket_counts[rocket_id] = 0
        rocket_counts[rocket_id] += 1
    
    # Fetch rocket names
    rocket_names = {}
    for rocket_id in rocket_counts.keys():
        rocket_info = requests.get(f'https://api.spacexdata.com/v4/rockets/{rocket_id}').json()
        rocket_names[rocket_id] = rocket_info.get('name')

    # Convert to a list of tuples (name, count)
    rocket_list = [(rocket_names[rid], count) for rid, count in rocket_counts.items()]

    # Sort by number of launches (descending), then by name (ascending)
    rocket_list.sort(key=lambda x: (-x[1], x[0]))

    # Print results
    for name, count in rocket_list:
        print(f"{name}: {count}")
