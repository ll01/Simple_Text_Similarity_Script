import main


if __name__ == "__main__":
    a = "hello my name is david."
    b = "I really like trains"

    # a = "Pure Siesta DAB/ FM Digital Alarm"
    # b = "Pure Siesta Mi Series 2 DAB+/FM Alarm Clock Radio "

    albert  = main.ALBERT(a, b)
    print(albert.score)
