## Audio Sentiment Analysis Pipeline (ASAP)
An end-to-end Python pipeline for performing sentiment analysis on audio files of call-center conversations.

## Installation
Directions for installing and running the pipeline.

### Required Libraries
- [pyAudioAnalysis](https://github.com/tyiannak/pyAudioAnalysis): Refer to [this suggested commit](https://github.com/tyiannak/pyAudioAnalysis/pull/15/commits/6b75b6716a7dbd90abb6ee0ecc613f8fb3e8f575) to make changes to audioSegmentation.py in the library so that the library may be properly imported.
- [openSMILE](http://audeering.com/technology/opensmile/): Follow directions from tutorial so that openSMILE can be run from the command line with `SMILExtract`.
- [Lasagne](http://lasagne.readthedocs.io/en/latest/user/installation.html) is required for ASAP with an LSTM. With a Lasagne install you should also get Theano.
- [pydub](https://github.com/jiaaro/pydub)
- [hmmlearn](https://github.com/hmmlearn/hmmlearn)

### ASAP
Once the above libraries are installed just clone this repo to get ASAP.

## Using ASAP
- NOTE: Unfortunately, pyAudioAnalysis doesn't support python 3.0+ so the pipeline must be run with python v2.7.
- To run the whole pipeline end-to-end run `AudioSentimentPipeline.py`. Running `AudioSentimentPipeline.py -h` will give a description of the input requirements and options.
- The input to ASAP is a CSV where the first column is the name of the audio file and the last column is the label. Each row corresponds to a different input file. Currently the input CSV must be in the same directory as all of the input audio files.
- Each step of the pipeline can be run separately. Run `process_raw_data.py -h`, `classify.py -h`, and `lstm.py -h` for how to use each individually.
- Refer to openSMILE documentation for how to change the feature extraction. Config files are in the [opensmile_conf](https://github.com/avewells/audio_sentiment_analysis/tree/master/audio_sentiment_analysis/opensmile_conf) folder. Default extraction uses IS09 features. To change from full file features to sliding window features edit [FrameModeFunctionals.conf.inc](https://github.com/avewells/audio_sentiment_analysis/blob/master/audio_sentiment_analysis/opensmile_conf/shared/FrameModeFunctionals.conf.inc).

### Example
This example uses files that are not from call centers so do not contain two speakers and use fake labels just to show everything working.
`python AudioSentimentPipeline.py -i data/input.csv -o outputs/ --hmm`
