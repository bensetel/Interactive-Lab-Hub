import pandas as pd
import glob
import re
#['pl_151.txt', 'right_shoulder.txt', 'right_wrist.txt', 'pl_101.txt', 'pl_1.txt', 'left_elbow.txt', 
def main():
    fn = glob.glob('*.txt')
    header = 'x,y,z'
    for name in fn:
        fi = open(name, 'r')
        lines = fi.readlines()
        fi.close()
        newfn = re.sub('.txt', '.csv', name)
        csv_fi = open(newfn, 'w+')
        csv_fi.write(header + '\n')
        for line in lines:
            if not('(' in line):
                continue
            s = line.split('(')[1]
            s = s.split(', visibility')[0]
            s = re.sub('[a-z]=', '', s)
            s = re.sub(' ', '', s)
            newline = s + '\n'
            csv_fi.write(newline)
        csv_fi.close()


