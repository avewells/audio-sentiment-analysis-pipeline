'''
Processes raw audio files so that they may be used with a sentiment analysis model. Can split each
audio file into smaller chunks based on speaker and pauses and perform feature extraction.

Accepts following input format:
A CSV file in which the first column contains the name of an audio file and the last column
contains the label associated with that particular file

Intermediate audio segments and extracted features will be saved for future use.

Author: Avery Wells 2017
'''

import sys
import os
import argparse
import glob
import subprocess
import numpy as np
import pandas as pd
from pyAudioAnalysis import audioSegmentation as aS
from pyAudioAnalysis import audioBasicIO as aIO
from pydub import AudioSegment
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def split_call_into_speakers(call_file, out_loc):
    '''
    Attempts to split a call file into different segments each time the speaker changes using
    speaker diarization. This method assumes there are two speakers in the file (sales and customer)
    and will cut out dial tones and any receptionists before the two speakers' conversation.
    '''
    # set output directories
    no_rings_out_dir = os.path.join(out_loc, 'calls_no_ringtones')
    if not os.path.exists(no_rings_out_dir):
        os.makedirs(no_rings_out_dir)
    diarized_out_dir = os.path.join(out_loc, 'calls_split_by_speaker')
    if not os.path.exists(diarized_out_dir):
        os.makedirs(diarized_out_dir)

    # load in raw audio file
    print(call_file)
    raw_audio = AudioSegment.from_file(call_file, 'wav')
    file_name = os.path.splitext(os.path.basename(call_file))[0]

    # uses trained HMM to determine where the ringtones are and only use audio from after
    # last detected ring and exports intermediate file
    curr_path = os.path.dirname(os.path.realpath(__file__))
    ring_labels = aS.hmmSegmentation(call_file, os.path.join(curr_path, 'hmmRingDetect'), False)
    segs, flags = aS.flags2segs(ring_labels[0], 1.0) # 1.0 is the mid-term window step from above model
    no_rings_audio = raw_audio[segs[-1, 0]*1000:segs[-1, 1]*1000]
    temp_out_loc = os.path.join(no_rings_out_dir, file_name) + '.wav'
    no_rings_audio.export(temp_out_loc, format='wav')

    # split on speakers now setting num speakers to 2
    diarized = aS.speakerDiarization(temp_out_loc, 2, mtSize=0.5, mtStep=0.1)

    # determine which label was given to customer and salesperson
    cust = diarized[0]

    # output the segments
    no_rings_audio = AudioSegment.from_file(temp_out_loc, format='wav') # update segment so indexing is right
    segs, flags = aS.flags2segs(diarized, 0.1)  #mtstep from above
    curr_call_out_base = os.path.join(diarized_out_dir, file_name)
    if not os.path.exists(curr_call_out_base):
        os.makedirs(curr_call_out_base)
    for seg in range(segs.shape[0]):
        # skip segments shorter than 1s (usually 'um' or something)
        if segs[seg, 1] - segs[seg, 0] < 1:
            continue
        out_seg = no_rings_audio[segs[seg, 0]*1000:segs[seg, 1]*1000]
        if flags[seg] == cust:
            out_seg.export(os.path.join(curr_call_out_base, str(seg) + '_cust.wav'), format='wav')
        else:
            out_seg.export(os.path.join(curr_call_out_base, str(seg) + '_sales.wav'), format='wav')


def extract_audio_features(in_csv, csv_loc, out_loc, split_flag):
    '''
    Uses openSMILE to extract multiple low level audio features. Features are extracted
    in batch. These features are the same features used in INTERSPEECH 2009 Emotion
    Challenge. Results in 384 dimensional feature vector for each file. Labels are added
    as last column.
    '''
    # openSMILE settings for use later
    curr_path = os.path.dirname(os.path.realpath(__file__))
    config = os.path.join(curr_path, 'opensmile_conf/IS09_emotion.conf')
    negcsv = os.path.join(out_loc, 'negfeatures.csv')
    poscsv = os.path.join(out_loc, 'posfeatures.csv')
    
    # if wavs were split by above function then have to do slightly different process
    if split_flag:
        split_out_dir = 'calls_split_by_speaker'
        for idx, row in in_csv.iterrows():
            if row[0].lower().endswith('.wav'):
                split_call_folder = os.path.join(out_loc, split_out_dir,
                                                 os.path.splitext(os.path.basename(row[0]))[0])
            else:
                split_call_folder = os.path.join(out_loc, split_out_dir, row[0])
            for audio_chunk in glob.glob(os.path.join(split_call_folder, '*.wav')):
                # split features into pos and neg CSVs so that labels can be easily added
                # and so looping through the files again isn't necessary
                if row[-1] == 0:
                    csvout = negcsv
                else:
                    csvout = poscsv
                opensmile = subprocess.Popen(['SMILExtract', '-C', config, '-instname', row[0], '-I',
                                              audio_chunk, '-csvoutput', csvout, '-timestampcsv', '0'],
                                             stdout=subprocess.PIPE)
                opensmile.communicate()
    else:
        # extract features from files in same directory as input CSV
        for idx, row in in_csv.iterrows():
            if row[0].lower().endswith('.wav'):
                input_call = row[0]
            else:
                input_call = row[0] + '.wav'
            call_path = os.path.join(os.path.dirname(csv_loc), input_call)
            if row[-1] == 0:
                csvout = negcsv
            else:
                csvout = poscsv
            opensmile = subprocess.Popen(['SMILExtract', '-C', config, '-instname', input_call, '-I',
                                          call_path, '-csvoutput', csvout, '-timestampcsv', '0'],
                                         stdout=subprocess.PIPE)
            opensmile.communicate()

    # add labels to csv (openSMILE uses semi colon instead of comma for some reason)
    neg_features = pd.read_csv(negcsv, sep=';')
    pos_features = pd.read_csv(poscsv, sep=';')
    neg_features['label'] = np.zeros((neg_features.shape[0], 1))
    pos_features['label'] = np.ones((pos_features.shape[0], 1))

    # export combined features
    combined_features = neg_features.append(pos_features)
    combined_features.to_csv(os.path.join(out_loc, 'features.csv'), index=False, header=None)



def process_csv_input(csv_loc, out_loc, split_flag, extract_flag):
    '''
    Will split and/or extract features from audio files designated by a CSV file.
    When splitting files, each smaller chunk is given the same label as the whole file.
    If splitting and extracting then features are extracted from each chunk of the audio file
    instead of the whole file.
    '''
    in_csv = pd.read_csv(csv_loc)
    # check if need to do both splitting and extraction
    if split_flag and extract_flag:
        print('Splitting provided audio files into chunks and then extracting features.')
        for idx, row in in_csv.iterrows():
            call_path = os.path.join(os.path.dirname(csv_loc), row[0])
            if row[0].lower().endswith('.wav'):
                split_call_into_speakers(call_path, out_loc)
            else:
                split_call_into_speakers(call_path + '.wav', out_loc)
        extract_audio_features(in_csv, csv_loc, out_loc, split_flag)
    elif split_flag:
        print('Splitting provided audio files into chunks.')
        for idx, row in in_csv.iterrows():
            call_path = os.path.join(os.path.dirname(csv_loc), row[0])
            if row[0].lower().endswith('.wav'):
                split_call_into_speakers(call_path, out_loc)
            else:
                split_call_into_speakers(call_path + '.wav', out_loc)
        extract_audio_features(in_csv, csv_loc, out_loc, split_flag)
    elif extract_flag:
        print('Extracting features from provided audio files.')
        extract_audio_features(in_csv, csv_loc, out_loc, split_flag)


def main(args, pipe=False):
    '''
    Checks passed arguments and performs requested actions.
    '''
    if not pipe:
        parser = argparse.ArgumentParser(description='Process raw audio files for sentiment analysis.')
        parser.add_argument('-i', '--input', dest='csv_loc', required=True,
                            help='Path to CSV file where first column is path to audio file and last column is label.')
        parser.add_argument('-o', '--out', dest='out_loc', required=True,
                            help='Path to where the audio chunks and feature file should be saved.')
        parser.add_argument('-s', '--split', dest='split_flag', action='store_true', default=False,
                            help='If each input file should be split into chunks by speaker.')
        parser.add_argument('-e', '--extract', dest='extract_flag', action='store_true', default=False,
                            help='If feature extraction should be performed on the input samples')
        args = parser.parse_args()

    # make sure they supplied either -s or -e
    if args.split_flag or args.extract_flag:
        # csv was given
        if args.csv_loc:
            process_csv_input(args.csv_loc, args.out_loc, args.split_flag, args.extract_flag)
        # no inputs given
        else:
            sys.exit('Must provide a CSV where first column is path to audio files and last is label.')
    else:
        sys.exit('Must provide either -s or -e or both.')


if __name__ == '__main__':
    main(sys.argv[1:])
