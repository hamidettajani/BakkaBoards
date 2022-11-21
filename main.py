import random
import pandas

user = input("Skriv inn brukenavnet ditt: ")
exampleScore = random.randint(0,100)

with open("leaderboard.csv", "a") as leaderboard:
    leaderboard.write(f"{user}, {exampleScore}\n")
    leaderboard.close()


with open("leaderboard.csv", "r") as file:
    data = pandas.read_csv(file)
    print("\n")
    print(data.sort_values(by="Score"))
