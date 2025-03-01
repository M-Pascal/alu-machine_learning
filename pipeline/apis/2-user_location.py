#!/usr/bin/env python3
"""
Fetches and prints the location of a GitHub user using the GitHub API.

Usage:
    ./2-user_location.py <GitHub API User URL>

Example:
    ./2-user_location.py https://api.github.com/users/holbertonschool

- Prints the user's location if found.
- Prints 'Not found' if the user doesn't exist.
- If the API rate limit is exceeded (403 status),prints the time until reset.
"""

import requests
import sys
import time

def get_user_location(api_url):
    try:
        response = requests.get(api_url)
        
        if response.status_code == 403:  # Rate limit exceeded
            reset_time = int(response.headers.get('X-Ratelimit-Reset', 0))
            wait_minutes = round((reset_time - time.time()) / 60)
            # Fix: Used `.format()` instead of f-string
            print("Reset in {} min".format(wait_minutes))
            return
        
        data = response.json()
        print(data.get('location', 'Not found'))
        
    except requests.RequestException:
        print('Not found')

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: ./2-user_location.py <GitHub API User URL>")
        sys.exit(1)
    
    get_user_location(sys.argv[1])
