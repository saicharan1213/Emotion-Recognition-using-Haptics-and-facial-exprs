import numpy as np
import json
import warnings
from tensorflow import keras
from PIL import Image
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
import pandas as pd


warnings.filterwarnings("ignore")
model = keras.models.load_model('emotion_detection_cnn_model.h5')
expression_true = ['angry','happy','sad']
with open('touch_recog_model_ann.pkl', 'rb') as f:
    touch_model = pickle.load(f)



name = []
for i in range(0,3):
    for j in range(0,2):
        k=j
        name.append(str(i)+'_'+str(k+1))   


x_t = pd.read_csv('haptic_samples.csv')
y_t = pd.read_csv('target_haptics.csv')
x_t = x_t.drop(columns = ['Unnamed: 0.1'])
y_t = y_t.drop(columns = ['Unnamed: 0'])

face_exp = 0
intent = ""

json_dataset = """{
   "intents":[
      {
         "tag":"depressed",
         "responses":[
            "everything will be alright",
            "why are you depressed"
         ]
      },
      {
         "tag":"request",
         "responses":[
            "I can certainly do that for you",
            "Sure, here you go"
         ]
      },
      {
         "tag":"caring",
         "responses":[
            "you are such a caring person",
            "i like the way you treat me"
         ]
      },
      {
         "tag":"sarcastic",
         "responses":[
            "stop being sarcastic"
         ]
      },
      {
         "tag":"annoyed",
         "responses":[
            "why you annoyed?",
            "I am sorry for annoying you"
         ]
      },
      {
         "tag":"hit",
         "responses":[
            "what wrong did i do",
            "please dont hit me"
         ]
      },
      {
         "tag":"surprise",
         "responses":[
            "I Think you are excited",
            "uhh you are surprised ok but dont hit me hard"
         ]
      },
      {
         "tag":"tickle",
         "responses":[
            "I havent laughed this hard",
            "uhh stop, my stomach hurts"
         ]
      },
      {
         "tag":"console",
         "responses":[
            "thanks for being there for me",
            "thank you, I appreciate it"
         ]
      }
   ]
}"""
dataset = json.loads(json_dataset)

for i in range(len(name)):
    file = "test/"+ name[i] + ".jpg"
    im = Image.open(file) 
    pred = np.array(im)
    pred = np.expand_dims(pred, axis=0)
    
    face_exp = np.argmax(model.predict(pred))
    touch = touch_model.predict(np.array(x_t.iloc[20+i]).reshape(1,-1))
    
    print("actual expression is "+expression_true[int(name[i][0])]+" actual touch "+y_t.iloc[20+i][0]+"\n")
    if face_exp ==0:
        expression = 'angry'
    elif face_exp ==1:
        expression = 'happy'
    elif face_exp ==2:
        expression = 'sad'
    print("pred expression is "+str(expression)+" pred touch "+str(touch[0])+"\n")
    if (touch == 'tap')  and expression=='sad':
        intent = 'depressed'
    elif (touch == 'smooth') and expression=='sad':
        intent = 'console'
    elif (touch == 'scratch') and expression=='sad':
        intent = 'request'
    elif (touch == 'scratch') and expression=='happy':
        intent = 'tickle'
    elif (touch == 'tap') and expression=='happy': 
        intent = 'surprise'
    elif (touch == 'smooth') and expression=='happy':
        intent = 'caring'
    elif (touch == 'tap') and expression=='angry':
        intent = 'hit'
    elif (touch == 'scratch') and expression=='angry':
        intent = 'annoyed'
    elif (touch == 'smooth') and expression=='angry':
        intent = 'sarcastic'
    else:
        intent = ""
    print("intent is ",intent)
    for intent1 in dataset['intents']:
        if intent1['tag']==intent:
            lst = intent1['responses']   
            n = len(intent1['responses'])
            rnum = np.random.randint(n)   ## choose a response from the list at random
            print('Chatbot response', lst[rnum], '\n')
    print("--------------------------------------------------")