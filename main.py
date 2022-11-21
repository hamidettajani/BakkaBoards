import random
import pandas as pd

L_Board = "leaderboard.csv"

with open(L_Board, "a") as leaderboard:
    user = input("Skriv inn brukenavnet ditt: ")
    exampleScore = random.randint(0,100)

    leaderboard.write(f"{user}, {exampleScore}\n")
    leaderboard.close()


with open(L_Board, "r") as file:
    data = pd.read_csv(file)
    print("\n")
    print(data.sort_values(by="Score"))
