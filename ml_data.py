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

ERROR = []


def get_prev_fixtures(fixtures, fixture_id, team_id, n=1):
    fixtures["fixture.date"] = pd.to_datetime(fixtures["fixture.date"])
    fixture = fixtures[fixtures["fixture.id"] == fixture_id]
    date = fixture["fixture.date"].iloc[0]

    before_fixtures = fixtures[
        ((fixtures["teams.home.id"] == team_id) | (fixtures["teams.away.id"] == team_id)) &
        (fixtures["fixture.date"] < date)
    ]

    # Sort by date in descending order
    before_fixtures = before_fixtures.sort_values(by="fixture.date", ascending=False)

    if before_fixtures.shape[0] > 0:
        return before_fixtures.head(n)["fixture.id"]

    return None


def get_fixture_stats(fixture_stats, fixture_id, team_id, home):
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

    if home:
        stats.columns = [f"{col} (home)" for col in stats.columns]
    else:
        stats.columns = [f"{col} (away)" for col in stats.columns]

    return stats


def create_data_and_targets(fixtures, fixture_stats):
    # TODO: optimize?
    data = pd.DataFrame()
    targets = pd.Series(name="winner", dtype=int)

    for (i, row) in tqdm(fixtures.iterrows(), desc="Creating data and targets", total=fixtures.shape[0]):
        home = get_prev_fixtures(fixtures, row["fixture.id"], row["teams.home.id"])
        away = get_prev_fixtures(fixtures, row["fixture.id"], row["teams.away.id"])

        if home is None or away is None:
            continue

        # TODO: als er meer dan 1 wedstrijd is
        home_stats = get_fixture_stats(fixture_stats, home.iloc[0], row["teams.home.id"], True)
        away_stats = get_fixture_stats(fixture_stats, away.iloc[0], row["teams.away.id"], False)

        if home_stats is None or away_stats is None:
            continue

        stats = pd.concat([home_stats, away_stats], axis=1)
        data = pd.concat([data, stats], axis=0)

        if row["teams.home.winner"] is True:
            targets = pd.concat([targets, pd.Series(1)])
        elif row["teams.away.winner"] is True:
            targets = pd.concat([targets, pd.Series(-1)])
        else:
            targets = pd.concat([targets, pd.Series(0)])

    # Reset the index
    data.index = range(data.shape[0])
    targets.index = range(targets.shape[0])

    return data, targets


def main():
    fixtures = pd.read_csv("data/fixtures.csv", low_memory=False)
    fixtures = fixtures[fixtures["fixture.status.short"].isin(["FT", "AET", "PEN"])]
    # fixtures = fixtures.head(500)

    fixture_stats = pd.read_csv("data/fixture_stats1-7809.csv", low_memory=False)

    data, targets = create_data_and_targets(fixtures, fixture_stats)
    data.to_csv("data/data.csv", index=False)
    targets.to_csv("data/targets.csv", index=False, header=False)

    if ERROR:
        with open("errors.txt", "w") as f:
            for error in ERROR:
                f.write(error + "\n")


if __name__ == "__main__":
    main()
