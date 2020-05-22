import use
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def start():
    a = "hello my name is david"
    b = "kevin is not very nice"
    
    # a = "Pure Siesta DAB/ FM Digital Alarm"
    # b = "Pure Siesta Mi Series 2 DAB/FM Alarm Clock Radio"


    use_model = use.USE()
    print (use_model.compare_texts(a,b))

    start()
if __name__ == "__main__":
    start()