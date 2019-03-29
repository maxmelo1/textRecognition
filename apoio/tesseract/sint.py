import pyttsx3

speak = pyttsx3.init('sapi5')

speak.say('ola mundo')
speak.runAndWait()
