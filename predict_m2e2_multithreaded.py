import argparse
import shutil
import json
from predict import predict, initialize_params_and_model, load_IC_model
import numpy as np
from exp import NumpyArrayEncoder
from multiprocessing import Process
import os
import glob
from tqdm import tqdm
import pickle

def swap_events(e2e_rel):
    temp_sent = e2e_rel['sent_1']
    temp_char_id = e2e_rel['e1_start_char']
    temp_e_mention = e2e_rel['e1_mention']
    e2e_rel['sent_1'] = e2e_rel['sent_2']
    e2e_rel['e1_start_char'] = e2e_rel['e2_start_char']
    e2e_rel['e1_mention'] = e2e_rel['e2_mention']
    e2e_rel['sent_2'] = temp_sent
    e2e_rel['e2_start_char'] = temp_char_id
    e2e_rel['e2_mention'] = temp_e_mention
    return e2e_rel

def predict_m2e2(data, save_file_prefix, gpu_num, sample_size, batch_size, error_file):
    params, model = initialize_params_and_model(gpu_num, batch_size)
    model = load_IC_model(model, gpu_num)

    batch_idxs = list(range(0,len(data), sample_size))
    if batch_idxs[-1] < len(data):
        batch_idxs.append(len(data))

    error_keys = []
    for i in tqdm(range(len(batch_idxs)-1)):
        data_list_orig = data[batch_idxs[i]:batch_idxs[i+1]]
        sen_events_list = []
        data_list = []
        for k, vs in data_list_orig:
            # Hack to remove advertisements events
            selected_vs = []
            for v in vs:
                if 'Follow Reuters' in v['sent_1'] or 'Follow Reuters' in v['sent_2']:
                    continue
                selected_vs.append(v)

            sen_events_list.extend(selected_vs)
            data_list.append((k, selected_vs))

        # try:
        event_preds = predict(model, params, sen_events_list, 'json', gpu_num, True)
        # except Exception as ee:
        #     print('Error during prediction: ', ee)
        #     error_keys.extend(list(dict(data_list).keys()))
        #     with open(error_file, 'wb') as f:
        #         pickle.dump(error_keys, f)
        #     continue
        event_preds = event_preds['array'][1:, :]

        event_id = 0
        output = {}
        for k, v in data_list:
            found_event_event_rel = []
            for event_event_rel in v:
                event_event_rel['confidence'] = event_preds[event_id, :]
                found_event_event_rel.append(event_event_rel)
                event_id += 1
            output[k] = found_event_event_rel

        with open(save_file_prefix + '_{}.json'.format(i), 'w') as f:
            json.dump(output, f, cls=NumpyArrayEncoder)

    
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", default='/home/hammad/kairos/Subevent_EventSeg/output/test_val_te2ve.json', help="input corpus file")
    parser.add_argument("--save_file", default='test_seseg.json', help="output file events in joint constrained learning format")
    parser.add_argument("--num_process_per_gpu", type=int, default=1, help="Number of processes per gpu")
    parser.add_argument("--gpus", type=str, default='7', help="exact gpu numbers")
    parser.add_argument("--num_gpu", type=int, default=1, help="Number of gpus")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--sample_size", type=int, default=1, help="Batch size")

    args = parser.parse_args()

    save_file = args.save_file
    error_file = args.save_file.split('.json')[0] + '_error.pk'

    already_extracted = {}
    if os.path.exists(save_file):
        with open(save_file) as f:
            already_extracted = json.load(f)
    
    error_keys = set()
    if os.path.exists(error_file):
        with open(error_file, 'rb') as f:
            error_keys.update(set(pickle.load(f)))

    # Save temp file for video2article in this directory
    temp_dir = args.save_file.split('.json')[0]
    os.makedirs(temp_dir, exist_ok=True)

    # Already extracted e2e rels in temp files
    for file in glob.glob(os.path.join(temp_dir, '*.json')):
        with open(file) as f:
            temp_data = json.load(f)

        already_extracted.update(temp_data)
    
    # Error keys in temp file
    for file in glob.glob(os.path.join(temp_dir, '*.pk')):
        with open(file, 'rb') as f:
            temp_error_keys = pickle.load(f)

        error_keys.update(set(temp_error_keys))

    # Save already done data
    with open(save_file, 'w') as f:
        json.dump(already_extracted, f)
    
    # Save error keys
    with open(error_file, 'wb') as f:
        pickle.dump(list(error_keys), f)

    # Exclude error or already done data from data to be processed
    with open(args.input_file) as f:
        in_data = json.load(f)

    print('Found already done: {}/{}'.format(len(already_extracted), len(in_data)))
    print('Found Error vids: {}/{}'.format(len(error_keys), len(in_data)))
    for key in already_extracted:
        try:
            del(in_data[key])
        except:
            continue

    for key in error_keys:
        try:
            del(in_data[key])
        except:
            continue
 
    in_data = list(in_data.items())
    gpus = args.gpus.split()
    total_data_divisions = args.num_gpu * args.num_process_per_gpu

    vid_split_idxs = np.linspace(0, len(in_data), num=total_data_divisions, endpoint=False, dtype=int)
    vid_split_idxs = np.unique(vid_split_idxs)

    vid_splits = []
    i=0
    for i in range(1, vid_split_idxs.shape[0]):
        vid_splits.append(in_data[vid_split_idxs[i-1]:vid_split_idxs[i]])

    vid_splits.append(in_data[vid_split_idxs[i]:])

    processV = []
    i = 0
    for gpu_id in range(args.num_gpu):
        gpu_no = gpus[gpu_id]
        for _ in range(args.num_process_per_gpu):
            save_file_prefix = os.path.join(temp_dir, str(i))
            error_file_worker = os.path.join(temp_dir, str(i) + '_error.pk')
            processV.append(Process(target=predict_m2e2, args = (vid_splits[i], save_file_prefix, int(gpu_no), args.sample_size, 
                                                                    args.batch_size, error_file_worker)))
            i += 1

    for i in range(len(vid_splits)):
        processV[i].start()

    for i in range(len(vid_splits)):
        processV[i].join()

    print('Joining temporary files')
    e2e_joined = {}
    if os.path.exists(save_file):
        with open(save_file) as f:
            e2e_joined = json.load(f)

    for file in glob.glob(os.path.join(temp_dir, '*.json')):
        with open(file) as f:
            data = json.load(f)

        e2e_joined.update(data)

    print('Joining temporary error files')
    for file in glob.glob(os.path.join(temp_dir, '*.pk')):
        with open(file, 'rb') as f:
            data = pickle.load(f)

        error_keys.update(data)

    print('Joined temporary files. Removing temp files.')
    shutil.rmtree(temp_dir)
    with open(save_file, 'w') as f:
        json.dump(e2e_joined, f, cls=NumpyArrayEncoder)

    with open(error_file, 'wb') as f:
        pickle.dump(list(error_keys), f)