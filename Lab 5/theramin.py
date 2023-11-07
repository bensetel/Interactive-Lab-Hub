import cv2
import time
import numpy as np
import HandTrackingModule as htm
import math
from ctypes import cast, POINTER
import alsaaudio
import subprocess

from mingus.midi import fluidsynth
import mingus.core.notes as notes
from mingus.containers import NoteContainer
from mingus.containers import Note
from mingus.containers import Bar

def play_audio():
    command = ['./loop_audio.sh']
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print('Looping audio playback - press q to quit')
    return process  # Return the process object

# def play_note(note_num):
#     print(int(note_num))
#     print('\n')
#     fluidsynth.play_Note(note_num,0,100)

def display_note(img, note_num, minX, maxX, R, G, B):
    cv2.rectangle(img, (minX, 150), (maxX, 400), (0, 0, 0), 3)
    cv2.rectangle(img, (minX, 400 - int((int(note_num)/95 * 250))), (maxX, 400), (R, G, B), cv2.FILLED)

    


def main():
    m = alsaaudio.Mixer(control='Speaker', cardindex=3)
    m.setvolume(100) 
    s = fluidsynth.init('/usr/share/sounds/sf2/FluidR3_GM.sf2',"alsa")
    #fluidsynth.set_instrument(0, 1) #piano on channel 0
    fluidsynth.set_instrument(1, 1) #48) #strings on channel 1
    #fluidsynth.set_instrument(1, 61) #brass section on channel 2
    fluidsynth.main_volume(1, 127)
    fluidsynth.control_change(1, 64, 127)
    

    ################################
    wCam, hCam = 640, 480
    ################################

    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)
    pTime = 0

    detector = htm.handDetector(detectionCon=int(0.7))
    minVol = 0
    maxVol = 100
    vol = 0
    volBar = 400
    volPer = 0

    piano_min = 20
    piano_max = 80
    strings_min = 0
    strings_max = 107
    brass_min = 0
    brass_max = 107

    start = time.time()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)
        if len(lmList) != 0:

            thumbX, thumbY = lmList[4][1], lmList[4][2] #thumb
            pointerX, pointerY = lmList[8][1], lmList[8][2] #pointer

            middleX, middleY = lmList[12][1], lmList[12][2]
            ringX, ringY = lmList[16][1], lmList[16][2]
            pinkyX, pinkyY = lmList[20][1], lmList[20][2]

            cx, cy = (thumbX + pointerX) // 2, (thumbY + pointerY) // 2

            #thumb
            # cv2.circle(img, (thumbX, thumbY), 15, (0, 0, 0), cv2.FILLED)

            #pointer
            cv2.circle(img, (pointerX, pointerY), 15, (255, 0, 0), cv2.FILLED)

            #middle
            cv2.circle(img, (middleX, middleY), 15, (0, 0, 255), cv2.FILLED)

            #ring
            # cv2.circle(img, (ringX, ringY), 15, (0, 0, 0), cv2.FILLED)

            #pinky
            cv2.circle(img, (pinkyX, pinkyY), 15, (0, 255, 0), cv2.FILLED)
            
            
            
            note_1 = Note().from_int(int(np.interp(pointerY, [50, 300], [piano_max, piano_min])))
            note_2 = Note().from_int(int(np.interp(middleY, [50, 300], [piano_max, piano_min])))
            note_3 = Note().from_int(int(np.interp(pinkyY, [50, 300], [piano_max, piano_min])))

            print('index note: ' + str(int(note_1)) + ' middle note: ' + str(int(note_2)) + ' pinky note: ' + str(int(note_3)))
            print('\n')


            #display pointer
            display_note(img, note_1, 30, 45, 255, 0, 0)

            #display middle
            display_note(img, note_2, 45, 60, 0, 0, 255)

            #display pinky
            display_note(img, note_3, 60, 75, 0, 255, 0)

            c = Bar('C', (4, 4))
            c.place_notes(note_1, 4)
            c.place_notes(note_2, 4)
            c.place_notes(note_3, 4)
            c.place_rest(4)
            fluidsynth.play_Bar(c, 1, bpm=200)
            

            
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        # cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX,
        #             1, (255, 0, 0), 3)

        cv2.imshow("Img", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            audio_process.terminate()  #
            break

    cap.release()
    cv2.destroyAllWindows()



#main()
