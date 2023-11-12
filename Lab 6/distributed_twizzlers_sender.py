import time
import board
import busio
import adafruit_mpr121
import ssl

import paho.mqtt.client as mqtt
import uuid
import adafruit_ssd1306
from adafruit_apds9960.apds9960 import APDS9960
import digitalio
import adafruit_ssd1306


client = mqtt.Client(str(uuid.uuid1()))
client.tls_set(cert_reqs=ssl.CERT_NONE)
client.username_pw_set('idd', 'device@theFarm')

client.connect(
    'farlab.infosci.cornell.edu',
    port=8883)

topic = 'IDD/cool_table/color_sensor'

i2c = busio.I2C(board.SCL, board.SDA)

mpr121 = adafruit_mpr121.MPR121(i2c)
apds = APDS9960(board.I2C()) #gesture
apds.enable_proximity = True 


while True:
    """
    for i in range(12):
        if mpr121[i].value:
        	val = f'Twizzler {i} touched!'
        	print(val)
        	client.publish(topic, val)
    """
    val = 'proximity is:' + str(apds.proximity)
    print(val)
    client.publish(topic, val)

    time.sleep(0.25)
