import pyttsx3

en = pyttsx3.init()
en.setProperty('rate', 150)
en.setProperty('volume', 1.0)

def speak_text(text: str):
    en.say(text)
    en.runAndWait()
