import pandas as pd
import sqlite3

conn = sqlite3.connect("soccer.sqlite")

def get_avg_team_attributes():
    query = """
    SELECT
        A.team_api_id,
        T.team_long_name,
        AVG(A.buildUpPlaySpeed) AS buildUpPlaySpeed,
        AVG(A.buildUpPlayPassing) AS buildUpPlayPassing,
        AVG(A.chanceCreationPassing) AS chanceCreationPassing,
        AVG(A.chanceCreationCrossing) AS chanceCreationCrossing,
        AVG(A.chanceCreationShooting) AS chanceCreationShooting,
        AVG(A.defencePressure) AS defencePressure,
        AVG(A.defenceAggression) AS defenceAggression,
        AVG(A.defenceTeamWidth) AS defenceTeamWidth
    FROM Team_Attributes AS A
    LEFT JOIN Team AS T ON T.team_api_id = A.team_api_id
    GROUP BY A.team_api_id
    """

    return pd.read_sql_query(query, conn)

df = get_avg_team_attributes()

df.to_sql("Avg_Team_Attributes", conn, if_exists="replace")

    
