import os
os.chdir('/home/ben/Interactive-Lab-Hub/Lab 2/mobility-database-catalogs')
from tools.operations import *
import requests
from pathlib import Path



TIMETABLE_STORAGE = '/home/ben/Interactive-Lab-Hub/Lab 2/timetables'


def main():
    all_cities = get_sources()
    city_list = ['Athens', 'Atlanta', 'Baltimore', 'Bangkok', 'Bilbao', 'Birmingham', 'Boston', 'Brisbane', 'Bruxelles', 'Canberra', 'Cannes', 'Chicago', 'Dallas', 'Denver', 'Detroit', 'Dublin', 'Firenze', 'Fukuoka', 'Genoa', 'Helsinki', 'Honolulu', 'Houston', 'Kampala', 'Kraków', 'Kōbe', 'Las Vegas', 'London', 'Los Angeles', 'Madrid', 'Manila', 'Melbourne', 'Miami', 'Milan', 'Montreal', 'Munich', 'Nairobi', 'Napoli', 'Paris', 'Perth', 'Philadelphia', 'Rio de Janeiro', 'Rome', 'San Francisco', 'Seattle', 'São Paulo', 'Toronto', 'Vancouver']
    selected_entries = fetch_cities(all_cities, city_list)
    for entry in selected_entries:
        scrape_timetables(entry)

#city_entires = [md(x) for x in list(md) if md(x)['location']['municipality'] in city_list]

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


def scrape_timetables(entry):
    try:
        print('='*40)
        print('now getting:', entry['provider'])
        my_url = entry['urls']['direct_download']
        r = requests.get(my_url)
        file_prefix = str(entry['mdb_source_id']) + '_' + entry['location']['country_code'] + '_' + entry['location']['municipality']
        file_name = my_url.split('/')[-1].split('.zip')[0] + '.zip'
        #file_name = str(entry['mdb_source_id']) + '.zip'
        #final_name = f'{TIMETABLE_STORAGE}/{file_name}'
        final_name = f'{TIMETABLE_STORAGE}/{file_prefix}_{file_name}'
        print('final_name:', final_name)
        with open(final_name, 'wb+') as f:
            f.write(r.content)

        fin_split = final_name.split('_')[0] #a little kluge but we have to preserve the directory name with the space instead of the backslash to test if it exists, since python insistts on the space version instead of the slash version, while bash (and thus os.system) insists on backslash version
        final_name = final_name.replace(" ", "\\ ")
        temp = TIMETABLE_STORAGE.replace(" ", "\\ ")
        new_dir = f"{temp}/{str(entry['mdb_source_id'])}"
        if os.path.exists(fin_split):
            os.system(f'rm -r {new_dir}') #get rid of outdated data before we unpack the new data
        os.system(f'mkdir {new_dir}')
        os.system(f'unzip {final_name} -d {new_dir}')
        os.system(f'rm {final_name}')
        
    except KeyError as e:
        print('Key Error!')
        print('entry:', entry)
        print('error:', e)


        
