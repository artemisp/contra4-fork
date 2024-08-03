"""
 Copyright (c) 2024, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import json
import requests
import pandas as pd
import freesound
import ast
import time
import gzip
from bs4 import BeautifulSoup
from tqdm import tqdm

# data2url = json.load(open('data/data2url.json'))

def load_gzipped_json_from_url(url):
    try:
        # Send a GET request to the URL
        response = requests.get(url)
        response.raise_for_status()  # Raises an HTTPError for bad responses

        # Decompress the content
        compressed_content = response.content
        decompressed_content = gzip.decompress(compressed_content)

        # Parse the JSON data
        data = json.loads(decompressed_content)
        return data
    except requests.RequestException as e:
        print(f"HTTP request error: {e}")
        return None
    except gzip.BadGzipFile as e:
        print(f"Error decompressing data: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None
    
def check_video_availability(url):
    time.sleep(1)
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            # Check for common phrases used when a video is unavailable
            if any(keyword in str(soup) for keyword in ["Video unavailable", "This video is not available"]):
                print("unavailable")
                return False
            return True
        else:
            print("Error checking video:", response.status_code)
            return False
    except Exception as e:
        print(f"Error checking video: {e}")
        return False

client = freesound.FreesoundClient()
client.set_token("lGPrXlLAKsOdHY1dxRohFV32ZXlFdrGRfVvNvUIW", "token")
def get_freesound_url(sound_id):
    # Initialize the Freesound client with your API key
    time.sleep(5)

    try:
        # Retrieve the sound object using its ID
        sound = client.get_sound(sound_id)
        # Return the URL of the audio file
        return sound.previews.preview_hq_mp3  # You can also choose preview_lq_mp3 for a lower quality version
    except Exception as e:
        print("Error retrieving sound:", e)
        return None

def load_csv_from_url(url):
    # try:
    # Load the CSV file directly into a DataFrame
    df = pd.read_csv(url, encoding='ISO-8859-1')
    return df
    # except Exception as e:
    #     print(f"An error occurred while loading the CSV: {e}")
    #     return None

def load_json_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        return response.json()  # Returns the json content of the response
    except requests.RequestException as e:
        print(f"An error occurred: {e}")
        return None


# # dict_keys(['audiocaps_val', 'clothov2_instruct_val', 'clothov1_instruct_val', 'objaverse_val', 'densecap_test', 'densecap_valid', 'coco_val', 'charades_v1_test', 'msrvtt_test'])
data2path = json.load(open('data/data2path.json'))
data2url = {}

print("AUDIOCAPS....")
data2url['audiocaps_val'] = {}
youtube_format = "https://www.youtube.com/watch?v={}&start={}&end={}"
audiocaps_annotations = {str(d['audiocap_id']): d for i,d in load_csv_from_url('https://raw.githubusercontent.com/cdjkim/audiocaps/master/dataset/val.csv').iterrows()}

for id in tqdm(data2path['audiocaps_val']['key2path']):
    if check_video_availability(youtube_format.format(audiocaps_annotations[id]['youtube_id'], int(audiocaps_annotations[id]['start_time']), int(audiocaps_annotations[id]['start_time'])+11).split('&')[0]):
        data2url['audiocaps_val'][int(id)] = {"url": youtube_format.format(audiocaps_annotations[id]['youtube_id'], int(audiocaps_annotations[id]['start_time']), int(audiocaps_annotations[id]['start_time'])+11), "start_time": audiocaps_annotations[id]['start_time'], "end_time": audiocaps_annotations[id]['start_time']+11}


print("CLOTHOv1....")
data2url['clothov1_instruct_val'] = {}
clotho_annotations = {str(d['file_name']): d for i,d in load_csv_from_url('https://zenodo.org/records/3490684/files/clotho_metadata_evaluation.csv').iterrows()}
for id in data2path['clothov1_instruct_val']['key2path']:
    data2url['clothov1_instruct_val'][id] = {k: clotho_annotations[id][k] for k in ['sound_link', 'start_end_samples']}
for k,v in  data2url['clothov1_instruct_val'].items():
    if 'url' in v and v['url']!= 'NA':
        continue
    try:
        v['url'] = get_freesound_url(v['sound_link'].split("/")[-1])
    except:
        v['url'] = 'NA'
    if v['start_end_samples']:
        try:
            v['start_time'] = ast.literal_eval(v['start_end_samples'])[0]
            v['end_time'] = ast.literal_eval(v['start_end_samples'])[1]
        except:
            pass

print("CLOTHOv2...")           
data2url['clothov2_instruct_val'] = {}
clotho_annotations = {str(d['file_name']): d for i,d in load_csv_from_url('https://zenodo.org/records/4783391/files/clotho_metadata_validation.csv').iterrows()}
for id in data2path['clothov2_instruct_val']['key2path']:
    data2url['clothov2_instruct_val'][id] = {k: clotho_annotations[id][k] for k in ['sound_link', 'start_end_samples']}
for k,v in  data2url['clothov2_instruct_val'].items():
    if 'url' in v and v['url']!= 'NA':
        continue
    try:
        v['url'] = get_freesound_url(v['sound_link'].split("/")[-1])
    except:
        v['url'] = 'NA'
    if v['start_end_samples']:
        try:
            v['start_time'] = ast.literal_eval(v['start_end_samples'])[0]
            v['end_time'] = ast.literal_eval(v['start_end_samples'])[1]
        except:
            pass
            
data2url['objaverse_val'] = {}
print("OBJAVERSE....")
import objaverse
objaverse_annotations = objaverse.load_annotations(list(data2path['objaverse_val']['key2path'].keys()))
for id in data2path['objaverse_val']['key2path']:
    data2url['objaverse_val'][id] = objaverse_annotations[id]
    data2url['objaverse_val'][id]['url'] = data2url['objaverse_val'][id]['embedUrl']
    
print("MSRVTT....")
data2url['msrvtt_test'] = {}
# msrvtt_data = {d['video_id']: d for d in json.load(open('/Users/apanagopoulou/Downloads/videodatainfoITALIANO_2017_6513_497_100_TestValidati (1) 2.json'))['videos']}
msrvtt_data ={d['video_id']: d for d in load_gzipped_json_from_url('https://github.com/crux82/msr-vtt-it/raw/master/msr-vtt-it/videodatainfoITALIANO_2017_6513_497_100_TestValidati.json.gz')['videos']}
for id in tqdm(data2path['msrvtt_test']['key2path']):
    id = id.split('.')[0]
    if check_video_availability(youtube_format.format(msrvtt_data[id]['url'].split('v=')[-1], int(msrvtt_data[id]['start time']), int(msrvtt_data[id]['end time'])).split('&')[0]):
        try:
            data2url['msrvtt_test'][id+'.mp4'] =  {"url": youtube_format.format(msrvtt_data[id]['url'].split('v=')[-1], int(msrvtt_data[id]['start time']), int(msrvtt_data[id]['end time'])), "start_time": msrvtt_data[id]['start time'], "end time": msrvtt_data[id]['end time']}
        except:
            continue
data2url['charades_v1_test'] = {}
msrvtt_data = {d['video_id']: d for d in json.load(open('/Users/apanagopoulou/Downloads/videodatainfoITALIANO_2017_6513_497_100_TestValidati (1) 2.json'))['videos']}
for id in data2path['charades_v1_test']['key2path']:
    data2url['charades_v1_test'][id] =  {"url": youtube_format.format(msrvtt_data[id]['ur'].split('v=')[-1], msrvtt_data[id]['start time'], msrvtt_data[id]['end time']), "start_time": msrvtt_data[id]['start time'], "end time": audiocaps_annotations[id]['end time']}

print("COCO VAL....")
data2url['coco_val'] = {}
coco_url_format = "http://images.cocodataset.org/{}"
for id in data2path['coco_val']['key2path']:
    data2url['coco_val'][id] =  {"url": coco_url_format.format(id)}
    
data_public = {}
index = 1
data = json.load(open('data/discrn_balanced.json'))
for d in data:
    flag=False
    for i,e in enumerate(d["examples"]):
        if e['source'] == 'audiocaps_mm_caption_val':
            e['source'] = 'audiocaps_val'
        if e['source'] == 'objaverse_pointllm_val':
            e['source'] = 'objaverse_val'
        if e["source"] in data2url:
            try:
                e["url"] = data2url[e['source']][str(e["id"])]['url'].replace('watch?v=', 'embed/')
                e["modality"] = d['modalities'][i]
            except:
                flag=True
                break
        else:
            flag=True
            break
    if flag:
        e["url"] = None
    
json.dump(data, open('data/discrn_balanced.json', 'w'), indent=4)