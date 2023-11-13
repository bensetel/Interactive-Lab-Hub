import time
import board
import ssl

import paho.mqtt.client as mqtt
import uuid


from vosk import Model, KaldiRecognizer
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
rec = KaldiRecognizer(model, args.samplerate)
#q = queue.Queue()


os.system('cvlc --play-and-exit mi.mp3')
while True:
    partial = rec.PartialResult()
    print(partial)
    if KEYWORD in partial:
        os.system('cvlc --play-and-exit klaxon.mp3')
        val = partial
        client.publish(topic, val)

    time.sleep(0.25)
