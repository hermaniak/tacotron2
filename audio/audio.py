

import os.path
from numpy import array, ma
import aubio
import argparse

def audio_to_seq(audio_file, **kwargs):

    downsample = 1
    samplerate = 22050 // downsample
    win_s = 4096 // downsample # fft size
    hop_s = 1024  // downsample # hop size
    s = aubio.source(audio_file, samplerate, hop_s)
    samplerate = s.samplerate
    tolerance = 0.8
    #pitch("yin", win_s, hop_s, samplerate)
    pitch_o = aubio.pitch("yin", win_s, hop_s, samplerate)
    pitch_o.set_unit("midi")
    pitch_o.set_tolerance(tolerance)

    pitches = []
    confidences = []

    # total number of frames read
    total_frames = 0
    while True:
        samples, read = s()
        pitch = pitch_o(samples)[0]
        #pitch = int(round(pitch))
        confidence = pitch_o.get_confidence()
        #if confidence < 0.8: pitch = 0.
        #print("%f %f %f" % (total_frames / float(samplerate), pitch, confidence))
        if pitch < 100 or pitch > 30:
            pitches += [int(pitch)]
        confidences += [confidence]
        total_frames += read
        if read < hop_s: break

    skip = 1

    pitches = array(pitches[skip:])
    confidences = array(confidences[skip:])
    times = [t * hop_s for t in range(len(pitches))]
    #import pdb;pdb.set_trace()
    cleaned_pitches = pitches
    #cleaned_pitches = ma.masked_where(cleaned_pitches < 0, cleaned_pitches)
    #cleaned_pitches = ma.masked_where(cleaned_pitches > 120, cleaned_pitches)
    #cleaned_pitches = ma.masked_where(confidences < tolerance, cleaned_pitches)
    #cleaned_pitches = [ i for i in cleaned_pitches if i !=0 ]
    return cleaned_pitches


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='input audio file')
    parser.add_argument('-o', '--output', type=str, help='store seq file')
    parser.add_argument('--hparams', type=str,
                        required=False, help='comma separated name=value pairs')

    args = parser.parse_args()
    #hparams = create_hparams(args.hparams)

    seq = audio_to_seq(args.input)  
    print (seq)
  
