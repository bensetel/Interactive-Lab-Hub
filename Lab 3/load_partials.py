import re
import datetime


tf =  open('/home/ben/Interactive-Lab-Hub/Lab 3/speech-scripts/vosk_dump.txt', 'r')
mystr = tf.read()
tf.close()
last_partial = mystr.split('{')[-1].split(':')[-1].strip()
last_partial = re.sub(r'[^A-Za-z0-9 ]+', '', last_partial).strip()

tf =  open(f'/home/ben/Interactive-Lab-Hub/Lab 3/output_{datetime.datetime.now().strftime("%m_%d_%Y__%H_%M_%S")}.txt', 'w+')
tf.write(last_partial)
tf.close()
    
