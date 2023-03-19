import pyttsx3
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)
model = tf.keras.models.load_model('model.h5')

def decode_sentiment(score, include_neutral=True):
    HAPPY = "happy"
    SAD = "sad"
    NEUTRAL = "neutral"
    SENTIMENT_THRESHOLDS = (0.4, 0.7)
    if include_neutral:        
        label = NEUTRAL
        if score <= SENTIMENT_THRESHOLDS[0]:
            label = SAD
        elif score >= SENTIMENT_THRESHOLDS[1]:
            label = HAPPY
        return label
    else:
        return SAD if score < 0.5 else HAPPY
    
def predict(text, include_neutral=True):
    x_test = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=300)
    score = model.predict([x_test])[0]
    label = decode_sentiment(score, include_neutral=include_neutral)
    return label

def convert_to_audio(text, emotion=None, gender="female"):
    engine = pyttsx3.init()

    if emotion is None:
        emotion = predict(text)

    # Set voice properties based on emotion and gender
    if gender == "female":
        engine.setProperty('voice', 'HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0')
    else:  # male
        engine.setProperty('voice', 'HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_DAVID_11.0')

    # Adjust speech rate, volume, and pitch based on emotion
    if emotion == "happy":
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 1)
        engine.setProperty('pitch', 1.2)
    elif emotion == "sad":
        engine.setProperty('rate', 90)
        engine.setProperty('volume', 0.6)
        engine.setProperty('pitch', 0.7)
    else:
        engine.setProperty('rate', 120)
        engine.setProperty('volume', 0.9)
        engine.setProperty('pitch', 1.0)


    engine.say(text)
    engine.runAndWait()

text = input("Enter your text: ")
emotion = input("Enter the emotion (happy, sad, neutral) or leave empty to auto-detect: ")
gender = input("Enter the gender (male, female): ")
convert_to_audio(text, emotion if emotion else None, gender)
