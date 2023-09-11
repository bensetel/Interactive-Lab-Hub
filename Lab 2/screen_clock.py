import time
import subprocess
import digitalio
import board
from PIL import Image, ImageDraw, ImageFont
import adafruit_rgb_display.st7789 as st7789
from time import strftime, sleep
import datetime
import json
import requests
import sys
import subprocess


ri_station_id = "B06"
server_url = 'http://localhost:5000'



# Configuration for CS and DC pins (these are FeatherWing defaults on M0/M4):
cs_pin = digitalio.DigitalInOut(board.CE0)
dc_pin = digitalio.DigitalInOut(board.D25)
reset_pin = None

# Config for display baudrate (default max is 24mhz):
BAUDRATE = 64000000

# Setup SPI bus using hardware SPI:
spi = board.SPI()

# Create the ST7789 display:
disp = st7789.ST7789(
    spi,
    cs=cs_pin,
    dc=dc_pin,
    rst=reset_pin,
    baudrate=BAUDRATE,
    width=135,
    height=240,
    x_offset=53,
    y_offset=40,
)

# Create blank image for drawing.
# Make sure to create image with mode 'RGB' for full color.
height = disp.width  # we swap height/width to rotate it to landscape!
width = disp.height
image = Image.new("RGB", (width, height))
rotation = 90

# Get drawing object to draw on image.
draw = ImageDraw.Draw(image)

# Draw a black filled box to clear the image.
draw.rectangle((0, 0, width, height), outline=0, fill=(0, 0, 0))
disp.image(image, rotation)
# Draw some shapes.
# First define some constants to allow easy resizing of shapes.
padding = -2
top = padding
bottom = height - padding
# Move left to right keeping track of the current x position for drawing shapes.
x = 0

# Alternatively load a TTF font.  Make sure the .ttf font file is in the
# same directory as the python script!
# Some other nice fonts to try: http://www.dafont.com/bitmap.php
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)

# Turn on the backlight
backlight = digitalio.DigitalInOut(board.D22)
backlight.switch_to_output()
backlight.value = True

p = subprocess.Popen(['python', '/home/ben/Interactive-Lab-Hub/Lab 2/MTAPI/app.py'], cwd='/home/ben/Interactive-Lab-Hub/Lab 2/MTAPI')

time.sleep(5)
print('slept')


north_trains = []
south_trains = []
trains_passed = 0

def play_animation():
    draw.rectangle((0, 0, width, height), outline=0, fill=(0, 0, 255))

    print('todo') 

while True:
    # Draw a black filled box to clear the image.
    draw.rectangle((0, 0, width, height), outline=0, fill=(0, 150, 150))

    #TODO: Lab 2 part D work should be filled in here. You should be able to look in cli_clock.py and stats.py 
    trains = requests.get(f'{server_url}/by-id/{ri_station_id}').json()
    cur_north = trains['data'][0]['N']
    for train in cur_north:
        train_time = train['time'].split('-04:00')[0] #for whatever reason, all times have a '-4:00' at the end
        if not(train_time in north_trains):
            north_trains.append(train_time)
    cur_south = trains['data'][0]['S']
    for train in cur_south:
        train_time = train['time'].split('-04:00')[0] #for whatever reason, all times have a '-4:00' at the end
        if not(train_time in south_trains):
            south_trains.append(train_time)
    for train_time in north_trains:
        if datetime.datetime.now() > datetime.datetime.strptime(train_time, '%Y-%m-%dT%H:%M:%S'):
            trains_passed += 1
            play_animation()
            north_trains.remove(train_time)
    for train_time in south_trains:
        if datetime.datetime.now() > datetime.datetime.strptime(train_time, '%Y-%m-%dT%H:%M:%S'):
            trains_passed += 1
            play_animation()
            south_trains.remove(train_time)
    
    # Display image.
    disp.image(image, rotation)
    y = top
    draw.text((x, y), 'trains passed: ' + str(trains_passed), font=font, fill ="#FFFFFF")
    disp.image(image, rotation)
    time.sleep(10)


