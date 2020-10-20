from neural_network import HWRModel


def main():
    model = HWRModel()
    
    # create and save
    '''
    model.load_data()
    model.create()
    model.compile()
    model.train(1)
    model.save()
    print("Accuracy: ", model.get_accuracy())
    '''
    
    #load
    model.load_data()
    model.load()
    model.compile()
    print("Accuracy: ", model.get_accuracy())


main()
