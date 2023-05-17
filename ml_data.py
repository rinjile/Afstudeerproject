"""
Name: Rinji Le
Student ID: 13344552
Course: Afstudeerproject Bachelor Informatica
Date: TODO

Description:
TODO
"""

import pandas as pd
from tqdm import tqdm
import os

ERROR = []


def get_last_fixtures(fixtures, fixture, n):
    home_id = fixture["teams.home.id"]
    away_id = fixture["teams.away.id"]

    last_fixtures_home = fixtures[
        (fixtures["teams.home.id"] == home_id) | (fixtures["teams.away.id"] == home_id)
    ]
    last_fixtures_away = fixtures[
        (fixtures["teams.home.id"] == away_id) | (fixtures["teams.away.id"] == away_id)
        ]

    # Sort by date in descending order
    last_fixtures_home = last_fixtures_home.sort_values(by="fixture.date", ascending=False)
    last_fixtures_away = last_fixtures_away.sort_values(by="fixture.date", ascending=False)

    if last_fixtures_home.shape[0] < n or last_fixtures_away.shape[0] < n:
        return None, None

    return (last_fixtures_home.head(n)["fixture.id"].reset_index(drop=True),
            last_fixtures_away.head(n)["fixture.id"].reset_index(drop=True))


def get_last_h2h_fixtures(fixtures, fixture, n):
    home_id = fixture["teams.home.id"]
    away_id = fixture["teams.away.id"]

    last_fixtures = fixtures[
        ((fixtures["teams.home.id"] == home_id) & (fixtures["teams.away.id"] == away_id)) |
        ((fixtures["teams.home.id"] == away_id) & (fixtures["teams.away.id"] == home_id))
    ]

    # Sort by date in descending order
    last_fixtures = last_fixtures.sort_values(by="fixture.date", ascending=False)

    if last_fixtures.shape[0] < n:
        return None

    return last_fixtures.head(n)["fixture.id"].reset_index(drop=True)


def get_fixture_score(fixtures, fixture_id, description):
    fixture = fixtures[fixtures["fixture.id"] == fixture_id]
    fixture = fixture[["score.fulltime.home", "score.fulltime.away"]]
    fixture.columns = [f"{col} ({description})" for col in fixture.columns]

    return fixture.astype(int).reset_index(drop=True)


def get_fixture_stats(fixture_stats, fixture_id, team_id, description):
    stats = fixture_stats[
        (fixture_stats["fixture.id"] == fixture_id) &
        (fixture_stats["response.team.id"] == team_id)
    ]
    if stats.shape[0] == 0:
        ERROR.append(f"Could not find stats for fixture {fixture_id} and team {team_id}.")
        return None

    stats = stats[["type", "value"]].T
    stats.columns = stats.loc["type"]
    stats = stats.drop("type")
    stats = stats.fillna(0)

    # Convert percentages to floats
    stats["Ball Possession"] = stats["Ball Possession"].str.replace("%", "").astype(float) / 100

    if "Passes %" not in stats.columns:
        return None

    stats["Passes %"] = stats["Passes %"].str.replace("%", "").astype(float) / 100

    stats.columns = [f"{col} ({description})" for col in stats.columns]

    return stats.reset_index(drop=True)


def add_target(targets, home_winner, away_winner):
    if home_winner is True:
        return pd.concat([targets, pd.DataFrame({"home": [1], "draw": [0], "away": [0]})], axis=0)
    elif away_winner is True:
        return pd.concat([targets, pd.DataFrame({"home": [0], "draw": [0], "away": [1]})], axis=0)
    else:
        return pd.concat([targets, pd.DataFrame({"home": [0], "draw": [1], "away": [0]})], axis=0)


def create_data_and_targets(fixtures, fixture_stats, n=5):
    # TODO: optimize?
    data = pd.DataFrame()
    targets_result = pd.DataFrame(columns=["home", "draw", "away"], dtype=int)
    targets_score = pd.DataFrame(columns=["home", "away"], dtype=int)

    for (i, row) in tqdm(fixtures.iterrows(), desc="Creating data and targets", total=fixtures.shape[0]):
        home, away = get_last_fixtures(fixtures.head(i), row, n)
        h2h = get_last_h2h_fixtures(fixtures.head(i), row, n)

        if None in [home, away, h2h]:
            continue

        stats = pd.DataFrame()

        for j in range(n):
            home_score = get_fixture_score(fixtures, home.iloc[j], f"home {j + 1}")
            home_stats = get_fixture_stats(fixture_stats, home.iloc[j], row["teams.home.id"], f"home {j + 1}")

            away_score = get_fixture_score(fixtures, away.iloc[j], f"away {j + 1}")
            away_stats = get_fixture_stats(fixture_stats, away.iloc[j], row["teams.away.id"], f"away {j + 1}")

            h2h_score = get_fixture_score(fixtures, h2h.iloc[j], f"h2h {j + 1}")
            h2h_stats = get_fixture_stats(fixture_stats, h2h.iloc[j], row["teams.home.id"], f"h2h {j + 1}")

            if None in [home_stats, away_stats, h2h_stats]:
                stats = pd.DataFrame()
                break

            stats = pd.concat([stats, home_score, home_stats, away_score, away_stats, h2h_score, h2h_stats], axis=1)

        if stats.shape[0] == 0:
            continue

        data = pd.concat([data, stats], axis=0)
        targets_result = add_target(targets_result, row["teams.home.winner"], row["teams.away.winner"])
        targets_score = pd.concat([targets_score, pd.DataFrame({"home": [row["goals.home"]], "away": [row["goals.away"]]})], axis=0)

    return data.reset_index(drop=True), targets_result.reset_index(drop=True), targets_score.reset_index(drop=True).astype(int)


def main():
    fixtures = pd.read_csv("data/fixtures.csv", low_memory=False)
    # Only use the top 5 leagues
    fixtures = fixtures[fixtures["league.name"].isin(["Premier League", "La Liga", "Serie A", "Bundesliga", "Ligue 1"])]
    # TODO: ["FT", "AET", "PEN"]?
    fixtures = fixtures[fixtures["fixture.status.short"].isin(["FT"])]

    fixtures["fixture.date"] = pd.to_datetime(fixtures["fixture.date"])
    fixtures = fixtures.sort_values(by="fixture.date", ascending=True)
    # fixtures = fixtures.head(2000)

    fixture_stats = pd.read_csv("data/fixture_stats.csv", low_memory=False)

    data, targets_result, targets_score = create_data_and_targets(fixtures, fixture_stats)
    data.to_csv("data/ml_data.csv", index=False)
    targets_result.to_csv("data/ml_targets_result.csv", index=False, header=False)
    targets_score.to_csv("data/ml_targets_score.csv", index=False, header=False)

    if os.path.exists("errors.txt"):
        os.remove("errors.txt")

    if ERROR:
        with open("errors.txt", "w") as f:
            for error in ERROR:
                f.write(error + "\n")
        print("Errors written to errors.txt")


if __name__ == "__main__":
    main()
