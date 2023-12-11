import time
from adafruit_servokit import ServoKit

# Create a servo kit object with 16 channels
kit = ServoKit(channels=16, frequency=60)

# Depending on your servo make, the pulse width min and max may vary,
# you want these to be as small/large as possible without hitting the hard stop
# for max range. You'll have to tweak them as necessary to match the servos you have!
SERVOMIN = 150  # this is the 'minimum' pulse length count (out of 4096)
SERVOMAX = 600  # this is the 'maximum' pulse length count (out of 4096)

# Function to set the servo pulse length in seconds
def set_servo_pulse(n, angle):
    # Limit the angle to the valid range (0 to 180)
    angle = max(0, min(180, angle))

    if n in [2, 13, 14]:
        angle = 180 - angle

    kit.servo[n].angle = angle
    time.sleep(0.01)

# Function to initialize all servos to the 0 position
def initialize_servos():
    for servo_num in servo:
        set_servo_pulse(servo_num, 0)

# Function to move the head servo smoothly from startAngle to endAngle
def move_servos(servoNum, startAngle, endAngle):
    # Determine the direction of movement
    step = 1 if startAngle < endAngle else -1
    
    # Loop from startAngle to endAngle
    for angle in range(startAngle, endAngle, step):
        set_servo_pulse(servoNum, angle)
        time.sleep(0.0001)
    
    # Optionally, add a short delay at the end of the loop
    time.sleep(0.2)

servo = [0, 1, 2, 3, 12, 13, 14]

initialize_servos()

# Infinite loop
try:
    while True:

        for servo_num in servo:
            # Move from 0 to 180
            move_servos(servo_num, 0, 180)
            
            # Move from 180 to 0
            move_servos(servo_num, 180, 0)

except KeyboardInterrupt:
    # Handle keyboard interrupt (e.g., script manually closed)
    print("Script interrupted. Resetting servos to zero.")
    initialize_servos()
