#sudo systemctl stop mini-screen.service

import os
os.chdir('/home/ben/Interactive-Lab-Hub/Lab 2/mobility-database-catalogs')
from tools.operations import *
os.chdir('/home/ben/Interactive-Lab-Hub/Lab 2')
#TODO - unkluge that

import pandas as pd
import numpy as np
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

import pytz

#import gtfs_kit as gk


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

buttonA = digitalio.DigitalInOut(board.D23)
buttonB = digitalio.DigitalInOut(board.D24)
buttonA.switch_to_input()
buttonB.switch_to_input()

TT_ROOT = '/home/ben/Interactive-Lab-Hub/Lab 2/timetables'



def next_station(skn, stations):
    print('next_station called')
    if skn > (len(stations.keys())-1):
        skn = 0
    else:
        skn += 1
    cur_station_id = list(stations.keys())[skn]
    return skn, cur_station_id

def prev_station(skn, stations):
    print('prev_station called')
    if skn == 0:
        skn = len(stations.keys())-1
    else:
        skn -= 1
    cur_station_id = list(stations.keys())[skn]
    return skn, cur_station_id

def play_animation():
    draw.rectangle((0, 0, width, height), outline=0, fill=(0, 0, 255))
    print('todo') 



def main():
    cur_station_id = "B06"
    server_url = 'http://localhost:5000'
    station_data_path = '/home/ben/Interactive-Lab-Hub/Lab 2/MTAPI/data/stations.json'
    with open(station_data_path) as fp:
        station_json = json.load(fp)

    stations = {}
    for key in station_json.keys():
        cur = station_json[key]
        stations.update({cur['id']:cur['name']})
    #cur_station_id = "B06"
    station_key_number = 222
    p = subprocess.Popen(['python', '/home/ben/Interactive-Lab-Hub/Lab 2/MTAPI/app.py'], cwd='/home/ben/Interactive-Lab-Hub/Lab 2/MTAPI')
    sleep(5) #give the server time to startup before querying it


    
    north_trains = []
    south_trains = []
    trains_passed = 0
    loop_counter=  0
    city_list = ['New York', 'Athens', 'Atlanta', 'Baltimore', 'Bangkok', 'Bilbao', 'Birmingham', 'Boston', 'Brisbane', 'Bruxelles', 'Canberra', 'Cannes', 'Chicago', 'Dallas', 'Denver', 'Detroit', 'Dublin', 'Firenze', 'Fukuoka', 'Genoa', 'Helsinki', 'Honolulu', 'Houston', 'Kampala', 'Kraków', 'Kōbe', 'Las Vegas', 'London', 'Los Angeles', 'Madrid', 'Manila', 'Melbourne', 'Miami', 'Milan', 'Montreal', 'Munich', 'Nairobi', 'Napoli', 'Paris', 'Perth', 'Philadelphia', 'Rio de Janeiro', 'Rome', 'San Francisco', 'Seattle', 'São Paulo', 'Toronto', 'Vancouver']
    cur_city = city_list[0]
    both_pressed = False
    color_dict = {}
    for city in city_list:
        color_dict.update({city:np.random.randint(0,255, size=[3])})


    all_cities = get_sources()
    selected_entries = fetch_cities(all_cities, city_list)

    systems = ['nyc_subway'] #many cities have more than one transit system
    cur_system = systems[0]
    station = 0
    all_times = []
    row_counter = 0
    station_changed = False
    system_changed = False
    all_times = []
    stops = pd.DataFrame()
    stop_times = pd.DataFrame()
    trips = pd.DataFrame()
    calendar = pd.DataFrame()
    timezone = 'America/New York'
    one_pressed = False
    
    while True:
        station_changed = False
        system_changed = False
        go_forward_station = False
        if (not buttonB.value) and (not buttonA.value): #change cities #todo - figure out a better way to test holding down the buttons for 3sec
            if not both_pressed:
                sleep(3)
                both_pressed = True
                continue
            #else: implied by the continue
            if cur_city == city_list[-1]:
                cur_city = city_list[0]
            
            else:
                cur_city = city_list[city_list.index(cur_city)+1]
                
            if not(cur_city == "New York"):
                 systems = get_systems(cur_city, selected_entries)
                 cur_system = systems[0]
                 system_changed = True
            print('changing cities!')
            print('new city is:', cur_city)
            print('new system is:', cur_system)
        
        elif (not buttonA.value) and (buttonB.value):
            if not one_pressed:
                sleep(3)
                one_pressed=True
                continue
            #else: implied by the continue
            if not(cur_city == "New York"):
                if cur_system == systems[-1]:
                    cur_system = systems[0]
                else:
                    cur_system = systems[systems.index(cur_system)+1]
                system_changed = True
            print('changing systems!')
            print('new system is:', cur_system)
        if one_pressed:
            go_forward_station = True
        if system_changed:
            stops, stop_times, trips, calendar, timezone = update_dfs(cur_system)
            cur_station = get_first_station(stops)
            row_counter = 0
            station_changed = True
            
        both_pressed = False
        one_pressed = False
        draw.rectangle((0, 0, width, height), outline=0, fill=tuple(color_dict[cur_city])) #(0, 150, 150))
        if cur_city == "New York":
            if ((not buttonA.value) and buttonB.value) or go_forward_station:
                station_key_number, cur_station_id = next_station(station_key_number, stations)
                north_trains = []
                south_trains = []
                loop_counter = 0
            elif (not buttonB.value) and buttonA.value:
                station_key_number, cur_station_id = prev_station(station_key_number, stations)
                north_trains = []
                south_trains = []
                loop_counter = 0

            #TODO: Lab 2 part D work should be filled in here. You should be able to look in cli_clock.py and stats.py 
            trains = requests.get(f'{server_url}/by-id/{cur_station_id}').json()
            cur_north = trains['data'][0]['N']
            for train in cur_north:
                train_time = train['time'].split('-04:00')[0] #for whatever reason, all times have a '-4:00' at the end

                if not(train_time in north_trains) and (datetime.datetime.now() < datetime.datetime.strptime(train_time, '%Y-%m-%dT%H:%M:%S')):
                    north_trains.append(train_time)
            cur_south = trains['data'][0]['S']
            for train in cur_south:
                train_time = train['time'].split('-04:00')[0] #for whatever reason, all times have a '-4:00' at the end
                if not(train_time in south_trains) and (datetime.datetime.now() < datetime.datetime.strptime(train_time, '%Y-%m-%dT%H:%M:%S')):
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

            if loop_counter % 10 == 0:
                 print('--------------------------')
                 print(north_trains)
                 print(south_trains)
                 print('--------------------------')
            # Display image.
            disp.image(image, rotation)
            y = top
            station_text_one = 'Current Station:'
            draw.text((x, y), station_text_one, font=font, fill ="#FFFFFF")
            y += font.getsize(station_text_one)[1]
            station_text_two = stations[cur_station_id]
            draw.text((x, y), station_text_two, font=font, fill ="#FFFFFF")
            y += font.getsize(station_text_two)[1]
            tp_text = 'Trains Passed: ' + str(trains_passed)
            draw.text((x, y), tp_text, font=font, fill ="#FFFFFF")
            disp.image(image, rotation)
            sleep(1)
            loop_counter += 1

            
        else:
            if ((not buttonA.value) and buttonB.value) or go_forward_station: #go one station forward
                if row_counter < len(stops)-1:
                    row_counter += 1
                else:
                    row_counter = 0
                cur_station = stops.iloc[row_counter]
                station_changed = True
            elif (not buttonB.value) and buttonA.value: #go one station backward
                if row_counter > 0:
                    row_counter -= 1
                else:
                    row_counter = len(stops)-1
                cur_station = stops.iloc[row_counter]
                station_changed = True
                
            if station_changed:
                all_times = get_times(cur_station, stop_times, trips, timezone, calendar)

            for time in all_times:
                if datetime.datetime.now(pytz.timezone('America/New_York')) > time:
                    trains_passed += 1
                    play_animation()
                    all_times.remove(time)
                    
            city_name = cur_city
            system_name = cur_system['provider']
            station_name = cur_station['stop_name']

            
            disp.image(image, rotation)
            y = top
            city = 'Current City: ' + city_name
            system_static = 'Current System:'
            station_static = 'Current Station:'

            rgb_fill = tuple([255-x for x in color_dict[cur_city]]) #get the inverse color of our background for maximum readability
            hex_fill = '#%02x%02x%02x' % rgb_fill
            
            draw.text((x, y), city, font=font, fill =hex_fill)
            y += font.getsize(city)[1]

            draw.text((x, y), system_static, font=font, fill =hex_fill)
            y += font.getsize(system_static)[1]
            draw.text((x, y), system_name, font=font, fill =hex_fill)
            y += font.getsize(system_name)[1]

            draw.text((x, y), station_static, font=font, fill =hex_fill)
            y += font.getsize(station_static)[1]
            draw.text((x, y), station_name, font=font, fill =hex_fill)
            y += font.getsize(station_name)[1]
            
            tp_text = 'Trains Passed: ' + str(trains_passed)
            draw.text((x, y), tp_text, font=font, fill =hex_fill)

            
            disp.image(image, rotation)
            sleep(1)

def get_first_station(stops):
    return stops.iloc[0]

def update_dfs(system):
    dirpath = f"{TT_ROOT}/{str(system['mdb_source_id'])}"
    stops = pd.read_csv(f'{dirpath}/stops.txt', index_col='stop_id')
    stop_times = pd.read_csv(f'{dirpath}/stop_times.txt', index_col='trip_id')
    agency = pd.read_csv(f'{dirpath}/agency.txt')
    timezone = agency['agency_timezone'].values[0] #assume all systems within an agency are in the same timezone
    trips = pd.read_csv(f'{dirpath}/trips.txt')
    calendar = pd.read_csv(f'{dirpath}/calendar.txt')
    return stops, stop_times, trips, calendar, timezone

def get_times(cur_station, stop_times, trips, timezone, calendar):

    my_times = stop_times[stop_times['stop_id'] == cur_station.name]
    all_times = []
    for index, my_time in my_times.iterrows():
        add_a_day = False
        hour = int(my_time['departure_time'].split(':')[0])
        if hour > 23: #they use 25 etc for stops after midnight, and 24 instead of 00 for midnight itself
            add_a_day = True
        my_trip = trips[trips['trip_id'] == my_time.name] 
        my_cal = calendar[calendar['service_id']==my_trip['service_id'].values[0]]
        cur_weekday = datetime.datetime.now().strftime('%A').lower()
        if my_cal[cur_weekday].values[0] == 1: #if the service is running today
            tz = pytz.timezone(timezone)
            if add_a_day:
                to_fix = my_time['departure_time'].split(':')
                fixed = str(int(to_fix[0])-24) + ':' + to_fix[1] + ':' + to_fix[2]
                time_string = (datetime.datetime.now(tz) + datetime.timedelta(days=1)).strftime('%m-%d-%y') + ' ' + fixed
            else:
                time_string = datetime.datetime.now(tz).strftime('%m-%d-%y') + ' ' + my_time['departure_time']      
            local_time = tz.localize(datetime.datetime.strptime(time_string, '%m-%d-%y %H:%M:%S'))
            new_york_time = local_time.astimezone(pytz.timezone('America/New_York'))
            if new_york_time > datetime.datetime.now(pytz.timezone('America/New_York')): #if the trip hasn't already happened
                all_times.append(new_york_time)
    return all_times
            
    
            
def fetch_cities(md, cl):
    dupe_ids = [1248, 591, 339, 323, 210, 4, 2] #birmingham alabama instead of birmingham uk, london ON instead of london UK etc
    keys = list(md)
    entries = []
    for key in keys:
        try:
            if (md[key]['location']['municipality'] in cl) and not(md[key]['mdb_source_id'] in dupe_ids):
                entries.append(md[key])
        except KeyError as e:
             continue
    return entries

def get_systems(city, se):
    sys_list = []
    for entry in se:
        try:
            if entry['location']['municipality'] == city:
                sys_list.append(entry)
        except KeyError as e:
            print('key error!')
            print('entry:', entry)
            print('error:', e)
    return sys_list
