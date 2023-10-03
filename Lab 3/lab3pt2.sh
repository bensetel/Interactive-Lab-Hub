#from: https://elinux.org/RPi_Text_to_Speech_(Speech_Synthesis)#Festival_Text_to_Speech
#sudo systemctl stop mini-screen.service

echo "HI! Activate me by saying my key phrase. Activate" | festival --tts 

python3 ~/Interactive-Lab-Hub/Lab\ 3/speech-scripts/activationCode.py -m en -f ~/Interactive-Lab-Hub/Lab\ 3/speech-scripts/lab3pt2ACTIVATE.txt

echo "Tell me a color for me to display!" | festival --tts

python3 ~/Interactive-Lab-Hub/Lab\ 3/speech-scripts/color_changer.py -m en -f ~/Interactive-Lab-Hub/Lab\ 3/speech-scripts/lab3pt2COLOR.txt
