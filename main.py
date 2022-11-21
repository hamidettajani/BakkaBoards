import random

user = input("Skriv inn brukenavnet ditt: ")
exampleScore = random.randint(0,100)

with open("leaderboard.txt", "a") as leaderboard:
    leaderboard.write(f"{user} | Score: {exampleScore}\n")
    leaderboard.close()
