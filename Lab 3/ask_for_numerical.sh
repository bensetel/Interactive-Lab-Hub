#from: https://elinux.org/RPi_Text_to_Speech_(Speech_Synthesis)#Festival_Text_to_Speech

echo "please say a zip code" | festival --tts

python3 /home/ben/Interactive-Lab-Hub/Lab\ 3/speech-scripts/test_microphone.py -m en -f /home/ben/Interactive-Lab-Hub/Lab\ 3/speech-scripts/vosk_dump.txt

python3 /home/ben/Interactive-Lab-Hub/Lab\ 3/format_partial.py


