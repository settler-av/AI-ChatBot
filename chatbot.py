import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

# to load the model created in training
from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

# get the words and class from the pickle file
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# load the model
model = load_model('chatbot_model.h5')

# the model stores the numerical data we have to use it properly
# we need four different functions
# 1. to clean the sentences
# 2. function to get the bag of words
# 3. function to predict the class
# 4. function to get the response

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


def bag_of_word(sentence):
    sentence_words = clean_up_sentence(sentence) #
    bag = [0]*len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_word(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    result = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    result.sort(key = lambda x: x[1], reverse=True)
    return_list = []
    for i in result:
        return_list.append({'intent': classes[i[0]], 'probability': str(i[1])})
    return return_list

def get_responce(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


print("GO! bot is running!")
while True:
    print(get_responce(predict_class(input("")), intents))
