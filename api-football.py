"""
Name: Rinji Le
Student ID: 13344552
Course: Afstudeerproject Bachelor Informatica
Date: TODO

Description:
TODO
"""

import pandas as pd
import requests
import json
import sys
import os
from tqdm import tqdm
import time

URL = "https://v3.football.api-sports.io"
API_KEY = list(open(".api_key"))[0].strip()
HEADERS = {
    "x-apisports-key": API_KEY
}
IDS = {}


def read_ids(file):
    """
    Reads the ids and names from the given file and creates a dictionary.

    :param file: File with the ids and names.
    :return: Dictionary with the ids and names.
    """
    ids = {}

    for i, line in enumerate(list(open(file))):
        if i != 0:
            idx, name = line.strip().replace("\"", "").split(",")[:2]
            ids[name] = idx

    return ids


def request(path, query):
    """
    Sends a get request to the API. The API has a limit of 300 requests per
    minute, so the function sleeps for 0.2 seconds to prevent exceeding the limit.

    :param path: Path to the endpoint (str).
    :param query: Query parameters (dict).
    :return: Response from the API (requests.Response).
    """
    response = requests.request("GET", URL + path, headers=HEADERS, params=query)
    # TODO: 0.2
    time.sleep(6)  # Prevent exceeding API limit
    return response


def get_fixtures(start_season=2015, end_season=2021):
    """
    Gets the fixtures for all leagues in IDS for the given seasons.

    :param start_season: The first season to get the fixtures for (int).
    :param end_season: The last season to get the fixtures for (int).
    :return: All fixtures (pd.DataFrame).
    """
    fixtures = pd.DataFrame()

    for league in tqdm(IDS.keys(), desc="Leagues"):
        for season in tqdm(range(start_season, end_season + 1), leave=False, desc="Seasons"):
            query = {"league": IDS[league], "season": season}
            response = request("/fixtures", query).json()

            if response["errors"]:
                print(f"Error ({league}, {season}): {response['errors']}")
            else:
                new_fixtures = pd.json_normalize(response, record_path=["response"])
                fixtures = pd.concat([fixtures, new_fixtures])

    return fixtures


def save_fixtures(fixtures, filename):
    """
    Saves the fixtures to a json and csv file.

    :param fixtures: Dataframe with fixtures (pd.DataFrame).
    :param filename: Name of the file (str).
    :return: None.
    """
    fixtures.to_json(f"data/{filename}.json", orient="records", indent=4)
    fixtures.to_csv(f"data/{filename}.csv", index=False)


def main():
    if len(sys.argv) == 1:
        filename = "data"
    else:
        filename = sys.argv[1]

    if os.path.isfile(f"data/{filename}.json") or os.path.isfile(f"data/{filename}.csv"):
        print("File already exists!")
        while True:
            answer = input("Do you want to overwrite the file? (y/n): ")
            if answer == "y":
                break
            elif answer == "n":
                sys.exit()

    global IDS
    IDS = read_ids("ids.csv")

    fixtures = get_fixtures()
    save_fixtures(fixtures, filename)


if __name__ == "__main__":
    main()
