import json
from temp import extract_frames

import numpy as np
import json

from torch import div

with open('../JointConstrainedLearning/output/m2e2_e2e_rels_finer_all_events_with_confidence.json') as f:
    jcl = json.load(f)

with open('output/m2e2_rels_finer_all_events.json') as f:
    subev = json.load(f)

def swap_events(e2e_rel):
    temp_sent = e2e_rel['sent_1']
    temp_char_id = e2e_rel['e1_start_char']
    temp_e_mention = e2e_rel['e1_mention']
    temp_char_begin_id_relative_to_article = e2e_rel['e1_char_begin_id_relative_to_article']
    temp_char_end_id_relative_to_article = e2e_rel['e1_char_end_id_relative_to_article']
    e2e_rel['sent_1'] = e2e_rel['sent_2']
    e2e_rel['e1_start_char'] = e2e_rel['e2_start_char']
    e2e_rel['e1_mention'] = e2e_rel['e2_mention']
    e2e_rel['e1_char_begin_id_relative_to_article'] = e2e_rel['e2_char_begin_id_relative_to_article']
    e2e_rel['e1_char_end_id_relative_to_article'] = e2e_rel['e2_char_end_id_relative_to_article']
    e2e_rel['sent_2'] = temp_sent
    e2e_rel['e2_start_char'] = temp_char_id
    e2e_rel['e2_mention'] = temp_e_mention
    e2e_rel['e2_char_begin_id_relative_to_article'] = temp_char_begin_id_relative_to_article
    e2e_rel['e2_char_end_id_relative_to_article'] = temp_char_end_id_relative_to_article
    return e2e_rel

result = {}
for k, v1 in subev.items():
    v2 = jcl[k]
    selected = []
    for e1 in v1:
        c1 = e1['confidence']
        fl = 0
        for e2 in v2:
            if e2['e1_start_char'] == e1['e1_start_char'] and e2['e2_start_char'] == e1['e2_start_char'] and e1['sent_1'] == e2['sent_1'] and e1['sent_2'] == e2['sent_2']:
                c2 = e2['confidence']
                fl = 1
                break
        if fl == 0:
            raise ValueError

        cnet = (np.array(c1) + np.array(c2))/ 2
        pred = np.argmax(cnet)
        if pred == 0:
            selected.append(e1)
        elif pred == 1:
            selected.append(swap_events(e1))

    result[k] = selected

import numpy as np
def select_e2es(subev):
    result = {}
    for k, v1 in subev.items():
        selected = []
        for e1 in v1:
            c1 = e1['confidence']
            pred = np.argmax(c1)
            if pred == 0:
                selected.append(e1)
            elif pred == 1:
                selected.append(swap_events(e1))
        result[k] = selected
    return result


with open('output/jcl_subev_rels_finer_joint.json', 'w') as f:
    json.dump(result, f)
    
import numpy as np
def match_dicts(data, data_orig, eps):
    not_match = 0
    for k, vs in data.items():
            vos = data_orig[k]
            for v in vs:
                fl = 0
                for vo in vos:
                    if v['e1_start_char'] == vo['e1_start_char'] and v['e2_start_char'] == vo['e2_start_char'] and v['sent_1'] == vo['sent_1'] and v['sent_2'] == vo['sent_2']:
                        fl = 1
                        if np.any(np.absolute(np.array(v['confidence']) - np.array(vo['confidence'])) > eps):
                            not_match += 1
                        break
                if fl == 0:
                    raise ValueError
    return not_match

def check_a_sample_match_dicts(data, data_orig, eps):
    match_found = 0
    for k, vs in data.items():
            vos = data_orig[k]
            for v in vs:
                fl = 0
                for vo in vos:
                    if v['e1_start_char'] == vo['e1_start_char'] and v['e2_start_char'] == vo['e2_start_char'] and v['sent_1'] == vo['sent_1'] and v['sent_2'] == vo['sent_2']:
                        fl = 1
                        if np.all(np.absolute(np.array(v['confidence']) - np.array(vo['confidence'])) < eps):
                            print(v)
                            print(vo)
                            match_found = 1
                        break
                if fl == 0:
                    raise ValueError
                if match_found == 1:
                    break
            if match_found == 1:
                break
    return

import json
import glob
import os
def get_remaining_data_and_divide_it(data_file, done_data_files, temp_dir=None, div_parts=[]):
    with open(data_file) as f:
        data = json.load(f)
    done_data = {}
    for done_data_file in done_data_files:
        with open(done_data_file) as f:
            temp_data = json.load(f)
        done_data.update(temp_data)

    if temp_dir is not None:
        for file in glob.glob(os.path.join(temp_dir, '*.json')):
            with open(file) as f:
                temp_data = json.load(f)
            done_data.update(temp_data)
    
    rem_keys = set(data.keys()) - set(done_data.keys())
    rem_keys = list(rem_keys)

    sum_parts = sum(div_parts)
    div_ids = [0]
    for i in range(len(div_parts)-1):
        div_ids.append(int(len(rem_keys) * (div_parts[i] + div_ids)/sum_parts))
    
    rem_datas = []
    for i in range(len(div_ids)-1):
        keys_to_extract = rem_keys[div_ids[i]:div_ids[i+1]]
        rem_data = {}
        for k in keys_to_extract:
            rem_data[k] = data[k]
        rem_datas.append(rem_data)

    return done_data, rem_datas
    

done_videos=[]
videos_in_embed_dir = [v.split('.npy')[0] for v in videos_in_embed_dir]
for vid in tqdm(video_list):
    if vid in videos_in_embed_dir:
        num_frames = len(os.listdir(os.path.join(videos_frames_dir, vid)))
        with open(os.path.join(videos_embed_dir, vid + '.npy'), 'rb') as f:
            embeds = np.load(f)
            num_extracted_embeds = embeds.shape[0]
        if num_frames == num_extracted_embeds:
            done_videos.append(vid)
videos_to_embed = list(set(video_list) - set(done_videos))

def getNotNoneValKeys(d):
    key_list = []
    for k, vs in d.items():
            if len(vs) != 0:
                    key_list.append(k)
    return key_list


import numpy as np
def match_grounding_in_dicts(data, data_orig, eps):
    not_match = 0
    for k, vs in data.items():
            vos = data_orig[k]
            for v in vs:
                fl = 0
                for vo in vos:
                    if v['e1_start_char'] == vo['e1_start_char'] and v['e2_start_char'] == vo['e2_start_char'] and v['sent_1'] == vo['sent_1'] and v['sent_2'] == vo['sent_2']:
                        fl = 1
                        g1 = set([tuple(g) for g in v['grounded_segments']])
                        g2 = set([tuple(g) for g in vo['grounded_segments']])
                        if g1 != g2:
                            not_match += 1
                        break
                if fl == 0:
                    raise ValueError
    return not_match


test_small = {}
count_no_rel = 0
count_hier = 0
count_coref = 0
count_data = 0
for k, anots in data.items():
    if count_no_rel >= 20 and count_hier >= 40 and count_coref >= 40:
            print(count_no_rel, count_hier, count_coref)
            break
    selected_anots = []
    count_data += 1
    for anot in anots:
        label = anot['label']
        if label == 'NoRel' and count_no_rel >= 20:
            continue
        else:
            count_no_rel += 1
        if label == 'Hierarchical' and count_hier >= 40:
            continue
        else:
            count_hier += 1
        if label == 'Identical' and count_coref >= 40:
            continue
        else:
            count_coref += 1
        selected_anots.append(anot)
    if selected_anots:
        test_small[k] = selected_anots


count_no_rel = 0
count_hier = 0
count_coref = 0
for k, anots in test_small.items():
    for anot in anots:
        label = anot['label']
        if label == 'NoRel' :
            count_no_rel += 1
        if label == 'Hierarchical' :
            count_hier += 1
        if label == 'Identical' :
            count_coref += 1
