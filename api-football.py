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
import sys
import os
from tqdm import tqdm
import time

URL = "https://v3.football.api-sports.io"
API_KEY = list(open(".api_key"))[0].strip()
HEADERS = {"x-apisports-key": API_KEY}


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


def check_file_exists(filename):
    """
    Checks if the file already exists. If it does, the user is asked if he/she
    wants to overwrite the file.

    :param filename: Name of the file (str).
    :return: None.
    """
    if os.path.isfile(f"data/{filename}.json") or os.path.isfile(f"data/{filename}.csv"):
        print("File already exists!")
        while True:
            answer = input("Do you want to overwrite the file? (y/n): ")
            if answer == "y":
                break
            elif answer == "n":
                sys.exit(0)


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


def get_fixtures(ids, start_season=2015, end_season=2021):
    """
    Gets the fixtures for all leagues in IDS for the given seasons.

    :param ids: Dictionary with the ids and names of the leagues (dict).
    :param start_season: The first season to get the fixtures for (int).
    :param end_season: The last season to get the fixtures for (int).
    :return: All fixtures (pd.DataFrame).
    """
    fixtures = pd.DataFrame()

    for league in tqdm(ids.keys(), desc="Leagues"):
        for season in tqdm(range(start_season, end_season + 1), leave=False, desc="Seasons"):
            query = {"league": ids[league], "season": season}
            response = request("/fixtures", query).json()

            if response["errors"]:
                print(f"Error ({league}, {season}): {response['errors']}")
            else:
                new_fixtures = pd.json_normalize(response, record_path=["response"])
                fixtures = pd.concat([fixtures, new_fixtures])

    return fixtures


def save_data(data, filename):
    """
    Saves the data to a json and csv file.

    :param data: Dataframe with the data (pd.DataFrame).
    :param filename: Name of the file (str).
    :return: None.
    """
    data.to_json(f"data/{filename}.json", orient="records", indent=4)
    data.to_csv(f"data/{filename}.csv", index=False)


def get_fixture_stats():
    stats = pd.DataFrame()

    fixtures = pd.read_csv("data/fixtures.csv")
    fixture_ids = fixtures["fixture.id"]

    for idx in tqdm(fixture_ids, desc="Fixtures"):
        query = {"fixture": idx}
        response = request("/fixtures/statistics", query).json()

        if response["errors"]:
            print(f"Error (fixture {id}): {response['errors']}")
        else:
            new_stats = pd.json_normalize(response,
                                          record_path=["response", "statistics"],
                                          meta=[["parameters", "fixture"],
                                                ["response", "team", "id"],
                                                ["response", "team", "name"]])
            stats = pd.concat([stats, new_stats])

    return stats


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 api-football.py -1/-2 <filename>")
        sys.exit(1)

    filename = sys.argv[2]
    check_file_exists(filename)

    ids = read_ids("ids.csv")

    if sys.argv[1] == "-1":
        data = get_fixtures(ids)
    elif sys.argv[1] == "-2":
        data = get_fixture_stats()
    else:
        print("Usage: python3 api-football.py -1/-2 <filename>")
        sys.exit(1)

    save_data(data, filename)


if __name__ == "__main__":
    main()
