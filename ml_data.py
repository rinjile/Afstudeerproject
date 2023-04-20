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


def get_prev_fixtures(fixtures, fixture_id, team_id, n=10):
    fixtures["fixture.date"] = pd.to_datetime(fixtures["fixture.date"])
    fixture = fixtures[fixtures["fixture.id"] == fixture_id]
    date = fixture["fixture.date"].iloc[0]

    # TODO: met i als fixtures sorted is
    before_fixtures = fixtures[
        ((fixtures["teams.home.id"] == team_id) | (fixtures["teams.away.id"] == team_id)) &
        (fixtures["fixture.date"] < date)
    ]

    # Sort by date in descending order
    before_fixtures = before_fixtures.sort_values(by="fixture.date", ascending=False)

    if before_fixtures.shape[0] > 0:
        return before_fixtures.head(n)["fixture.id"].reset_index(drop=True)

    return None


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
    stats["Passes %"] = stats["Passes %"].str.replace("%", "").astype(float) / 100

    stats.columns = [f"{col} ({description})" for col in stats.columns]

    return stats.reset_index(drop=True)


def add_target(targets, home_winner, away_winner):
    if home_winner is True:
        return pd.concat([targets, pd.Series(1)])
    elif away_winner is True:
        return pd.concat([targets, pd.Series(-1)])
    else:
        return pd.concat([targets, pd.Series(0)])


def create_data_and_targets(fixtures, fixture_stats, n=10):
    # TODO: optimize?
    data = pd.DataFrame()
    targets = pd.Series(name="winner", dtype=int)

    for (_, row) in tqdm(fixtures.iterrows(), desc="Creating data and targets", total=fixtures.shape[0]):
        home = get_prev_fixtures(fixtures, row["fixture.id"], row["teams.home.id"], n=n)
        away = get_prev_fixtures(fixtures, row["fixture.id"], row["teams.away.id"], n=n)

        # TODO: missing data
        if home is None or away is None or home.size != n or away.size != n:
            continue

        stats = pd.DataFrame()

        for i in range(n):
            home_score = get_fixture_score(fixtures, home.iloc[i], f"home {i + 1}")
            away_score = get_fixture_score(fixtures, away.iloc[i], f"away {i + 1}")
            home_stats = get_fixture_stats(fixture_stats, home.iloc[i], row["teams.home.id"], f"home {i + 1}")
            away_stats = get_fixture_stats(fixture_stats, away.iloc[i], row["teams.away.id"], f"away {i + 1}")

            # TODO: missing data
            if home_stats is None or away_stats is None:
                stats = pd.DataFrame()
                break

            stats = pd.concat([stats, home_score, home_stats, away_score, away_stats], axis=1)

        if stats.shape[0] == 0:
            continue

        data = pd.concat([data, stats], axis=0)
        targets = add_target(targets, row["teams.home.winner"], row["teams.away.winner"])

    return data.reset_index(drop=True), targets.reset_index(drop=True)


def main():
    fixtures = pd.read_csv("data/fixtures.csv", low_memory=False)
    # Only use the top 5 leagues
    fixtures = fixtures[fixtures["league.name"].isin(["Premier League", "La Liga", "Serie A", "Bundesliga", "Ligue 1"])]
    # TODO: ["FT", "AET", "PEN"]?
    fixtures = fixtures[fixtures["fixture.status.short"].isin(["FT"])]

    # TODO: sort
    # fixtures["fixture.date"] = pd.to_datetime(fixtures["fixture.date"])
    # fixtures = fixtures.sort_values(by="fixture.date", ascending=True)
    fixtures = fixtures.head(200)

    fixture_stats = pd.read_csv("data/fixture_stats.csv", low_memory=False)

    data, targets = create_data_and_targets(fixtures, fixture_stats)
    data.to_csv("data/ml_data.csv", index=False)
    targets.to_csv("data/targets.csv", index=False, header=False)

    if os.path.exists("errors.txt"):
        os.remove("errors.txt")

    if ERROR:
        with open("errors.txt", "w") as f:
            for error in ERROR:
                f.write(error + "\n")
        print("Errors written to errors.txt")


if __name__ == "__main__":
    main()
