import json 
import pickle
import nltk
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


#Loading the Json File 

with open("chat.json") as file:
    bot = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words, labels, train, output = pickle.load(f)

except Exception as err:
    print(f"Data could not be read due to {err}")
    #Creating List to be used for iterations

    words = []
    labels = []
    intent_part_x = []
    intent_part_y = []


    #Iterating through the Json File and appending it to neccessary list 

    for intent in bot["intents"]:
        for pattern in intent["patterns"]:
            token = nltk.wordpunct_tokenize(pattern)   #Splitting the Pattern into tokens using NLTK 

            #Since token is a list no need of appending we just use the function extend

            words.extend(token)
            intent_part_x.append(token)
            intent_part_y.append(intent["tag"])


        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [ps.stem(w.lower()) for w in words]  #Stemming the word 

    words = sorted(list(set(words)))

    labels = sorted(labels) #Sorting the Label list Alpahbetically

    train = []
    output = []

    out_empty = [0 for _ in range(len(labels))]   #Creating a list of 0 to be used for the Bag of Words list

    for x, doc in enumerate(intent_part_x):
        bow = []

        token = [ps.stem(w) for w in doc]

        for w in words:
            if w in token:
                bow.append(1)
            else:
                bow.append(0)

        output_row = out_empty[:]
        output_row[labels.index(intent_part_y[x])] = 1

        train.append(bow)
        output.append(output_row)

    train = np.array(train)
    output = np.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, train, output), f)


try:
    model = load_model("model.h5")

except Exception as err:

    print(f'Error Message: {err}')


    #Depp Learning Nueral Net Layers to be used 

    model = Sequential()
    model.add(Dense(128, input_shape=(len(train[0]),), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(output[0]), activation='softmax'))

    # Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    #Fitting the model to the train and output for prediction 

    model.fit(train, output, epochs=200, batch_size=8, verbose=1)
    model.save("model.h5")    #Saving the model
#Functions for the Chat Bot

def bag_of_words(sentence, words):
    """
    Input:
    This functions create a bag of word for the sentence the user enters
    
    Output:
    It returns a numpy array to be used 
    Note: The Numpy array was reshaped to an array of 63 because the length of our train[0] is 63
    If you make changes or edit the json file to suit your needs then the Length of your train will change
    Do well to play around the codes, make changes to suit your need and understand how it works
    """
    bag = [0 for i, word in enumerate(words)]

    sentence_token = nltk.wordpunct_tokenize(sentence)
    sentence_token = [ps.stem(word.lower()) for word in sentence_token]

    for sentences in sentence_token:
        for i, word in enumerate(words):
            if word == sentences:
                bag[i] = 1
    
    return np.array(bag).reshape(-1, 63)

def chat():
    """
    Input:
    This is function that Activate the ChatBot for the User to Interact with it
    
    Output:
    It gives a Responses randomly.
    We selected a threshold to prevent it from giving out irrelevatnt response to the user
    You can quit talking with the Chat Bot by type quit in the chat Column
    
    Note: The Threshold was not selected Randomly, it was done after sereies of conversation with the bot to know,
    which threshold it starts giving out irrelevant responses if the responses is not in it database
    """
    print("Start Talking with the Bot,To Stop type quit")
    while True:
        inp = input("You :   ")
        if inp.lower() == "quit":
            break 

        input_data = [bag_of_words(inp, words)]
        results = model.predict(input_data)[0]
        results_index = np.argmax(results)
        tag = labels[results_index]
        
        if results[results_index] >= 0.844:

            #Looping through the json file

            for tags in bot["intents"]:
                if tags["tag"] == tag:
                    responses = tags["responses"]
            print(np.random.choice(responses))
        
        else:
            print("I dont quite Understand, Ask another question")

chat()
