import requests
from PIL import Image
import lenet
import matplotlib.pyplot as plt
import numpy as np
import cv2

def displayText():
    print("-- Training the Model --")
    print("Pick what you would like to test the model with")
    print("1. Turn Left")
    print("2. Speed Limit")
    print("3. Slippery Road")
    print("4. Yield")
    print("5. Bicycle Crossing")
    print("6. Paste own JPG URL of German Traffic Sign")
    print("Q. Quit")

def handleInput():
    userInput = input(">> ")
    match userInput:
        case "1":
            # turn left
            urlString = 'https://c8.alamy.com/comp/A0RX23/cars-and-automobiles-must-turn-left-ahead-sign-A0RX23.jpg'
        case "2":
            # speed limit
            urlString = 'https://c8.alamy.com/comp/G667W0/road-sign-speed-limit-30-kmh-zone-passau-bavaria-germany-G667W0.jpg'
        case "3":
            # slippery
            urlString = 'https://previews.123rf.com/images/bwylezich/bwylezich1608/bwylezich160800375/64914157-german-road-sign-slippery-road.jpg'
        case "4":
            # give way
            urlString = 'https://previews.123rf.com/images/pejo/pejo0907/pejo090700003/5155701-german-traffic-sign-no-205-give-way.jpg'
        case "5":
            # byclycle crossing
            urlString = 'https://c8.alamy.com/comp/J2MRAJ/german-road-sign-bicycles-crossing-J2MRAJ.jpg'
        case "6":
            print("The Image must be in JPG format")
            userUrl = input("Enter in URL >> ")
            urlString = userUrl
        case "q":
            exit(0)
        case _:
            urlString = ""
    return urlString

def handleImage(urlString):
    r = requests.get(urlString, stream=True)
    img = Image.open(r.raw)
    img.show()

def testModel(model, urlString):
    r = requests.get(urlString, stream=True)
    img = Image.open(r.raw)
    plt.imshow(img, cmap=plt.get_cmap('gray'))

    #Preprocess image
    img = np.asarray(img)
    img = cv2.resize(img, (32, 32))
    img = lenet.preprocessing(img)

    img = img.reshape(1, 32, 32, 1)

    predict = lenet.predictClass(model, img)
    signMap = getSignNames()
    predictString = signMap[int(predict[0])]
    #Test image
    print(f"\nPredicted sign: {predictString}\n")

def getSignNames():
    signMap = {0:"Speed Limit", 1 : "Speed Limit", 2 : "Speed Limit", 3 : "Speed Limit", 4 : "Speed Limit",
        5:"Speed Limit", 6 : "End of Speed Limit", 7:"Speed Limit", 8:"Speed Limit", 9:"No Passing",
        10:"No passing for vehicles over 3.5 metric tons", 11:"Right of way at next intersection",
        12:"Priority road", 13:"Yield", 14:"Stop", 15:"No Vehicles", 16:"Vehicles over 3.5 metric tons prohibites",
        17:"No entry",18:"General caution",19:"Dangerous curve to the left",20:"Dangerous curve to the right",21:"Double curve",
        22:"Bumpy road",23:"Slippery road",24:"Road narrows on the right",25:"Road work",26:"Traffic signals",
        27:"Pedestrians",28:"Children crossing",29:"Bicycles crossing",30:"Beware of ice/snow",31:"Wild animals crossing",
        32:"End of all speed and passing limits",33:"Turn right ahead",34:"Turn left ahead",35:"Ahead only",
        36:"Go straight or right",37:"Go straight or left",38:"Keep right",39:"Keep left",40:"Roundabout mandatory",
        41:"End of no passing",42:"End of no passing by vechiles over 3.5 metric tons"}
    return signMap

def testModelMain(model):
    runProgram = True
    while (runProgram):
        displayText()
        urlString = handleInput()
        handleImage(urlString)
        testModel(model, urlString)
