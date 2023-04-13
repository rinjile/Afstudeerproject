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
    if os.path.isfile(f"data/{filename}.csv"):
        print("File already exists!")
        while True:
            answer = input("Do you want to overwrite the file? (y/n): ")
            if answer == "y":
                break
            elif answer == "n":
                sys.exit(0)


def date_ago(date, years=0, months=0):
    """
    Calculates the date a certain amount of years and months ago.

    :param date: Date (str).
    :param years: Amount of years (int).
    :param months: Amount of months (int).
    :return: New date (str).
    """
    new_year = int(date[:4]) - years

    while int(date[5:7]) - months < 1:
        new_year -= 1
        months -= 12

    new_month = int(date[5:7]) - months

    return f"{new_year}-{new_month:02d}-{date[8:10]}"


def request(path, query, sleep=0.2):
    """
    Sends a get request to the API. The API has a limit of 300 requests per
    minute, so the function sleeps for 0.2 seconds to prevent exceeding the limit.

    :param path: Path to the endpoint (str).
    :param query: Query parameters (dict).
    :param sleep: Time to sleep (float).
    :return: Response from the API (requests.Response).
    """
    response = requests.request("GET", URL + path, headers=HEADERS, params=query)
    # sleep = 6  # TODO: verwijderen
    time.sleep(sleep)  # Prevent exceeding API's requests/min limit
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
                return fixtures
            else:
                new_fixtures = pd.json_normalize(response, record_path=["response"])
                fixtures = pd.concat([fixtures, new_fixtures])

    return fixtures


def get_fixture_stats():
    stats = pd.DataFrame()

    fixtures = pd.read_csv("data/fixtures.csv", low_memory=False)
    fixture_ids = fixtures["fixture.id"]

    for idx in tqdm(fixture_ids, desc="Fixture stats"):
        query = {"fixture": idx}
        response = request("/fixtures/statistics", query).json()

        if response["errors"]:
            print(f"Error (fixture {idx}): {response['errors']}")
            break
        else:
            new_stats = pd.json_normalize(response,
                                          record_path=["response", "statistics"],
                                          meta=[["response", "team", "id"],
                                                ["response", "team", "name"]])
            new_stats.insert(0, "fixture.id", idx)
            stats = pd.concat([stats, new_stats])

    return stats


def get_fixture_player_stats():
    player_stats = pd.DataFrame()

    fixtures = pd.read_csv("data/fixtures.csv", low_memory=False)
    fixture_ids = fixtures["fixture.id"]

    for idx in tqdm(fixture_ids, desc="Fixture player stats"):
        query = {"fixture": idx}
        response = request("/fixtures/players", query).json()

        if response["errors"]:
            print(f"Error (fixture {idx}): {response['errors']}")
            break
        else:
            new_player_stats = pd.json_normalize(response,
                                                 record_path=["response", "players", "statistics"],
                                                 meta=[["response", "players", "player", "id"],
                                                       ["response", "players", "player", "name"],
                                                       ["response", "team"]])
            new_player_stats.insert(0, "fixture.id", idx)
            player_stats = pd.concat([player_stats, new_player_stats])

    # Convert dict in "response.team" column to separate columns
    player_stats["team.id"] = player_stats["response.team"].apply(lambda x: x["id"])
    player_stats["team.name"] = player_stats["response.team"].apply(lambda x: x["name"])
    player_stats["team.update"] = player_stats["response.team"].apply(lambda x: x["update"])
    player_stats.drop(columns=["response.team"], inplace=True)

    # Move last 4 columns after the first column
    cols = player_stats.columns.tolist()
    cols = cols[:1] + cols[-4:] + cols[1:-4]
    player_stats = player_stats[cols]

    player_stats.columns = player_stats.columns.str.replace("response.players.", "", regex=False)

    return player_stats

def get_fixture_h2h():
    h2h = pd.DataFrame()

    fixtures = pd.read_csv("data/fixtures.csv", low_memory=False)
    fixtures = fixtures[["fixture.id", "teams.home.id", "teams.away.id", "fixture.date"]]

    for (idx, home, away, date) in tqdm(fixtures.itertuples(index=False), desc="Fixture h2h"):
        date = date[:10]  # Only use the date, not the time
        query = {
            "h2h": f"{home}-{away}",
            "from": date_ago(date, years=1),  # TODO: check
            "to": date,
            "status": "FT"
        }
        response = request("/fixtures/statistics", query).json()

        if response["errors"]:
            print(f"Error (fixture {idx}): {response['errors']}")
            break
        else:
            # TODO
            new_h2h = pd.json_normalize(response,
                                        record_path=[],
                                        meta=[])
            h2h = pd.concat([h2h, new_h2h])

    return h2h


def main():
    usage_message = "Usage: python3 api-football.py -1/-2/-3 <filename>"

    if len(sys.argv) != 3:
        print(usage_message)
        sys.exit(1)

    filename = sys.argv[2]
    check_file_exists(filename)

    ids = read_ids("data/ids.csv")

    if sys.argv[1] == "-1":
        data = get_fixtures(ids)
    elif sys.argv[1] == "-2":
        data = get_fixture_stats()
    elif sys.argv[1] == "-3":
        data = get_fixture_player_stats()
    else:
        print(usage_message)
        sys.exit(1)

    data.to_csv(f"data/{filename}.csv", index=False)


if __name__ == "__main__":
    main()
