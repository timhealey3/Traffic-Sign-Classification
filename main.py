import lenet
import test

def displayText():
    print("Welcome to Traffic Sign Classifier")
    user_input = input("Would you like to Train the Model? y/n: ")
    return user_input

def trainModel():
    model = lenet.modifiedModel()
    lenet.trainModel(model)
    return model

def testModel(model):
    test.testModelMain(model)

def main():
    program_running = True
    while(program_running):
        user_input = displayText()
        if (user_input == "y" or user_input == "Y"):
            model = trainModel()
            testModel(model)
            program_running = False
        if (user_input == "n" or user_input =="N"):
            program_running = False

main()
