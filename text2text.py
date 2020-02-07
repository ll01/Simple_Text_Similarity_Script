import main

def start():
    # a = "hello my name is david"
    # b = "welcome my name is david"
    
    a = "Pure Siesta DAB/ FM Digital Alarm"
    b = "Pure Siesta Mi Series 2 DAB/FM Alarm Clock Radio"

    model = main.USE()
    print (model.compare_texts(a,b))

    start()
if __name__ == "__main__":
    start()