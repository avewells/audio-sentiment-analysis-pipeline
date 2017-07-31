'''
Performs end-to-end sentiment analysis on call center conversation audio files.

Pipeline Steps:
1.) Remove any dial tones or receptionists from beginning of call
2.) Split call into chunks based on speaking turns using diarization
3.) Extract features from all chunks using OpenSMILE
4.) Classify sentiment using selected models

Inputs (and notes):
- CSV file in the same folder as all of the audio files to be analyzed. The
first column of the CSV should contain the name of the file and the last column
should contain the call's sentiment label (positive or negative).
- All audio files should be in WAV format. A helper mp3 to WAV conversion script
is included with this package.
- Labels in the CSV file should be 0 or 1.

Author: Avery Wells 2017
'''

import sys
import argparse
import os
from audio_sentiment_analysis import process_raw_data
from audio_sentiment_analysis import classify
from audio_sentiment_analysis import lstm


def main():
    '''
    Grab arguments for both pipeline scripts and pass them on. Will go all the way
    through the pipeline. For only splitting, extracting features, or classify use
    the other scripts individually
    '''
    # These checks are redundant with what is in other scripts but it is better
    # for the user to know they are missing arguments at the beginning rather 
    # than after splitting and extracting features
    parser = argparse.ArgumentParser(description='End-to-end audio sentiment analysis pipeline.')
    parser.add_argument('-i', '--input', dest='csv_loc', required=True,
                        help='Path to CSV file where first column is path to audio file and last column is label.')
    parser.add_argument('-o', '--out', dest='out_loc', required=True,
                        help='Path to where the audio chunks and feature file should be saved.')
    parser.add_argument('--hmm', dest='hmm_flag', action='store_true',
                        help='Classify with a Hidden Markov Model.')
    parser.add_argument('--rf', dest='rf_flag', action='store_true',
                        help='Classify with a random forest.')
    parser.add_argument('--lstm', dest='lstm_flag', action='store_true',
                        help='Classify with a LSTM.')
    parser.add_argument('--n_components', dest='n_components',
                        help='Number of components for the HMM.')
    parser.add_argument('--n_mix', dest='n_mix',
                        help='Number of Gaussian mixtures for the HMM.')
    parser.add_argument('--epochs', dest='num_epochs',
                        help='Number of training epochs.')
    parser.add_argument('--n_units', dest='num_units',
                        help='Number of LSTM units.')
    args = parser.parse_args()

    if args.hmm_flag or args.rf_flag or args.lstm_flag:
        args.split_flag = True
        args.extract_flag = True
        args.feat_loc = os.path.join(args.out_loc, 'features.csv')
        process_raw_data.main(args, pipe=True)
        if not args.lstm_flag:
            classify.main(args, pipe=True)
        else:
            lstm.main(args, pipe=True)
    else:
        sys.exit('Must choose at least one classification method. (--hmm, --rf, --lstm)')


if __name__ == '__main__':
    main()
