import os
import time
import board
import ssl

import paho.mqtt.client as mqtt
import uuid


from vosk import Model, KaldiRecognizer
import sounddevice as sd
import queue

client = mqtt.Client(str(uuid.uuid1()))
client.tls_set(cert_reqs=ssl.CERT_NONE)
client.username_pw_set('idd', 'device@theFarm')

client.connect(
    'farlab.infosci.cornell.edu',
    port=8883)

topic = 'IDD/cool_table/spi'
KEYWORD = 'ben'

model = Model(lang="en-us")
sr = sd.query_devices(kind='input')['default_samplerate']



os.system('cvlc --play-and-exit mi.mp3 &')
while True:
    with sd.RawInputStream(samplerate=sr, blocksize = 8000, device='default', dtype="int16", channels=1):
        rec = KaldiRecognizer(model, sr)
        partial = rec.PartialResult()
        print(partial)
        if KEYWORD in partial:
            os.system('cvlc --play-and-exit klaxon.mp3')
            val = partial
            client.publish(topic, val)

    time.sleep(0.25)
