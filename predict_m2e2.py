import argparse
import json
from predict import predict, initialize_params_and_model
import numpy as np
from exp import NumpyArrayEncoder


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

def predict_m2e2(data):
    data_list = data.items()
    sen_events_list = []
    for k, v in data_list:
        sen_events_list.extend(v)

    gpu_num = 7
    params, model = initialize_params_and_model(gpu_num)
    event_preds = predict(model, params, sen_events_list, 'json', gpu_num)
    event_preds = event_preds['array'][1:, :]

    event_id = 0
    output = {}
    for k, v in data_list:
        found_event_event_rel = []
        for event_event_rel in v:
            pred = np.argmax(event_preds[event_id, :])
            conf = event_preds[event_id, pred]
            # event_event_rel['confidence'] = conf
            # if pred == 0:
            #     found_event_event_rel.append(event_event_rel)
            # elif pred == 1:
            #     found_event_event_rel.append(swap_events(event_event_rel))
            event_event_rel['confidence'] = event_preds[event_id, :]
            found_event_event_rel.append(event_event_rel)
            event_id += 1
        output[k] = found_event_event_rel
    
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", default='../ETypeClus/outputs/m2e2_extracted_events_verbs_whole_corpus_w_objs_w_emention_h1.json', help="input corpus file")
    parser.add_argument("--save_file", default='output/temp.json', help="output file events in joint constrained learning format")
    args = parser.parse_args()

    with open(args.input_file) as f:
        in_data = json.load(f)

    out = predict_m2e2(in_data)

    with open(args.save_file, 'w') as f:
        json.dump(out, f, cls=NumpyArrayEncoder)