from logging import error
import cv2
import os
import argparse
from multiprocessing import Pool
import pickle
from tqdm import tqdm
import json
import math


EXTRACT_FPS = 3
def extract_frames(vid_name):
    error = []
    try:
        frame_dir = os.path.join(vid_frames_dir, vid_name)
        os.makedirs(frame_dir, exist_ok=True)

        vidcap = cv2.VideoCapture(os.path.join(vid_dir, vid_name + '.mp4'))
        count = 0
        while True:
            # To extract fps frames per second instead of all frames which can be huge
            vidcap.set(cv2.CAP_PROP_POS_MSEC,(int(count*1000 / EXTRACT_FPS)))
            success,image = vidcap.read()
            if not success:
                break
            cv2.imwrite(os.path.join(frame_dir, "%d.jpg" % (count+1)), image)     # save frame as JPEG file      
            count += 1

        to_return = ('success', vid_name)

    except Exception as ee:
        print(ee)
        to_return = ('error', vid_name)

    return to_return

def get_vid_len(vid):
    cap = cv2.VideoCapture(vid)
    fps = cap.get(cv2.CAP_PROP_FPS)      # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count/fps
    return duration

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vid_dir", dest="vid_dir", type=str, metavar='<str>', help="Dir path to load videos from")
    parser.add_argument("--vid_frames_dir", type=str, metavar='<str>', help="Dir path to save video frames")
    parser.add_argument("--vid_list", type=str, metavar='<str>', help="List of videos to extract frames from")
    parser.add_argument("--num_threads", dest="num_threads", type=int, help="Number of threads used to download articles")
    parser.add_argument("--extracted_vids", type=str, help="List of already extracted frames files")
    parser.add_argument("--error_file", type=str, help="List of already extracted frames files")
    args = parser.parse_args()

    with open(args.vid_list, 'rb') as f:
        video_list = pickle.load(f)

    vid_dir = args.vid_dir
    vid_frames_dir = args.vid_frames_dir

    error_vids = []
    if os.path.exists(args.error_file):
        with open(args.error_file, 'rb') as f:
            error_vids = pickle.load(f)

    print('Finding Done Videos')
    videos_to_download = []
    videos_in_frame_dir = os.listdir(args.vid_frames_dir)
    done_videos = []
    if os.path.exists(args.extracted_vids):
        with open(args.extracted_vids, 'rb') as f:
            done_videos = pickle.load(f)

    for vid in tqdm(video_list):
        if vid in done_videos or vid in error_vids:
            continue
        if vid in videos_in_frame_dir:
            num_extracted_frames = len(os.listdir(os.path.join(vid_frames_dir, vid)))
            num_theoretical_frames = get_vid_len(os.path.join(vid_dir, vid + '.mp4')) * 3
            if abs(num_extracted_frames - math.ceil(num_theoretical_frames)) < 3:
                done_videos.append(vid)
    videos_to_download = list((set(video_list) - set(done_videos)) - set(error_vids))

    print('Found done videos. Starting Frame extraction')
    with Pool(args.num_threads) as p:
        for result in tqdm(p.imap_unordered(extract_frames, videos_to_download), total=len(videos_to_download)):
            if result[0] == 'success':
                done_videos.append(result[1])
            elif result[0] == 'error':
                error_vids.extend(result[1])
            else:
                print('Value Error: Returned either succes nor error')
            if len(done_videos) % 1000 == 0:
                with open(args.extracted_vids, 'wb') as f:
                    pickle.dump(done_videos, f)
                with open(args.error_file, 'wb') as f:
                    pickle.dump(error_vids, f)