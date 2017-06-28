'''
Script to convert a directory of mp3s to WAVs. Utilizes pydub and FFMPEG library.

Usage: python mp3_to_wav.py INPUT_DIR OUT_DIR

If converting everything in a DIR, the files will retain their original names
and be placed in the OUT DIR if supplied. Otherwise in the INPUT DIR.
'''

import sys
import os
from pydub import AudioSegment

def convert_to_wav(inpath, inname, outdir):
    '''
    Converts the given mp3 file to wav.
    '''
    mp3 = AudioSegment.from_mp3(inpath)

    if outdir is None:
        # put in input dir
        basename = os.path.splitext(inpath)[0]
        mp3.export(basename + '.wav', format='wav')
    else:
        # put in output dir
        if os.path.isdir(outdir):
            basename = os.path.splitext(inname)[0]
            mp3.export(os.path.join(outdir, basename) + '.wav', format='wav')
        else:
            sys.exit('Make sure the output dirctory exists.')

# check inputs
if len(sys.argv) < 2:
    sys.exit('Must at least supply the input directory.\nUsage: python mp3_to_wav.py INPUT_DIR OUT_DIR')
elif len(sys.argv) < 3:
    INPUT_DIR = sys.argv[1]
    OUT_DIR = None
else:
    INPUT_DIR = sys.argv[1]
    OUT_DIR = sys.argv[2]

# check if it's a file or a directory to convert
if os.path.isdir(INPUT_DIR):
    print('Working...')
    for filename in os.listdir(INPUT_DIR):
        if filename.endswith('.mp3'):
            convert_to_wav(os.path.join(INPUT_DIR, filename), filename, OUT_DIR)
    print('Done.')
else:
    sys.exit('Make sure input directory exists.')
