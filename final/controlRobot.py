import time
from adafruit_servokit import ServoKit

import paho.mqtt.client as mqtt
import uuid
import queue
import ssl

#from detect import INTERNAL_DELIMITER, LINE_DELIMITER
INTERNAL_DELIMITER = '#'
LINE_DELIMITER = '*'

# Create a servo kit object with 16 channels
kit = ServoKit(channels=16, frequency=60)

# Depending on your servo make, the pulse width min and max may vary,
# you want these to be as small/large as possible without hitting the hard stop
# for max range. You'll have to tweak them as necessary to match the servos you have!
SERVOMIN = 150  # this is the 'minimum' pulse length count (out of 4096)
SERVOMAX = 600  # this is the 'maximum' pulse length count (out of 4096)

# Servo numbers for each servo at angle 0
servo = [0, 1, 2, 12, 13, 14]


"""
Servo num 0 is the left hand
Servo num 1 is the left elbow
Servo num 2 is the left shoulder

Servo num 14 is the right hand
Servo num 13 is the right elbow
Servo num 12 is the right shoulder

"""

servo_dict = {
    'left_hand':0,
    'left_elbow':1,
    'left_shoulder':2,
    'right_hand':14,
    'right_elbow':13,
    'right_shoulder':12,
}

global last_msg
last_msg = ''

# Function to initialize all servos to the 0 position
def initialize_servos():
    for servo_num in servo:
        set_servo_pulse(servo_num, 0)

def set_servo_pulse(n, angle):
    # Limit the angle to the valid range (0 to 180)
    angle = max(0, min(180, angle))
    if n in [2, 13, 14]:
        angle = 180 - angle
    kit.servo[n].angle = angle
    time.sleep(0.01)

def move_servos(servoNum, startAngle, endAngle):
    # Determine the direction of movement
    step = 1 if startAngle < endAngle else -1
    print(type(startAngle))
    print(type(endAngle))

    # Loop from startAngle to endAngle
    for angle in range(startAngle, endAngle, step):
        set_servo_pulse(servoNum, angle)
        time.sleep(0.0001)
        
def move_wrapper(servo_num, angle):
    start = round(kit.servo[servo_num].angle)
    move_servos(servo_num, start, angle)

def message_to_servo_angles(message):
    print('message to servo angles')
    print('message is:', message)
    
    landmark_msgs = message.split(LINE_DELIMITER)
    
    todo_list = [x.split('#') for x in landmark_msgs if x != '']
    retval = [[servo_dict[x[0]], int(x[1])] for x in todo_list]
    print('retval:', retval)
    return retval

def main():
    try:
        initialize_servos()
        time.sleep(0.05)
        client = mqtt.Client(str(uuid.uuid1()))
        client.tls_set(cert_reqs=ssl.CERT_NONE)
        client.username_pw_set('idd', 'device@theFarm')
        client.connect(
            'farlab.infosci.cornell.edu',
            port=8883)
        topic = 'IDD/cool_table/robit'
        client.subscribe(topic)
        client.on_message = on_message
        client.loop_forever()

    except KeyboardInterrupt:
        # Handle keyboard interrupt (e.g., script manually closed)
        print("Script interrupted. Resetting servos to zero.")
        
        # Reset all servos to zero
        initialize_servos()

def on_message(client, userdata, msg):
    global last_msg
    print('on message!')
    message = msg.payload.decode('UTF-8')
    if message == last_msg:
        print('== lastmsg')
    elif message == 'no_land':
        print('no landmarks!')
        set_zero()
        time.sleep(0.05)
    else:
        print('message is:', message)
        todo = message_to_servo_angles(message)
        print('todo is;', todo)
        for elem in todo:
            print('elem is:', elem)
            move_wrapper(elem[0], elem[1])
            print('called move_servo')
    time.sleep(0.5)
    
#shoulder rotation should be based on y distance of elbow from shoulder
#next servo out should be x distance from elbow to shoulder
        

if __name__=="__main__":
    main()
