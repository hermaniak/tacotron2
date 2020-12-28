

import os.path
import sys
from numpy import array, ma
import aubio
import argparse
import numpy as np
from scipy import stats

class Sequencer():

    #defaults:
    samplerate = 22050
    hop_s = 256
    win_s = 1024

    def __init__(self, **kwargs):
        if 'bpm' in kwargs and 'tpb' in kwargs and kwargs['bpm'] and kwargs['tpb']:
            self.hop_s = int(60 * self.samplerate / ( kwargs['bpm'] * kwargs['tpb'] ))
        elif 'hop_s' in kwargs:
            self.hop_s = kwargs['hop_s']
        else:
            self.hop_s = 256
        self.win_s = kwargs['win_s'] if 'win_s' in kwargs else 1024
            

    def to_seq(self, audio_file, **kwargs):
        #import pdb;pdb.set_trace()
        s = aubio.source(audio_file, self.samplerate, self.hop_s)
        tolerance = 0.8
        #pitch("yin", win_s, hop_s, samplerate)
        pitch_o = aubio.pitch("yin", self.win_s, self.hop_s, s.samplerate)
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
            if read < self.hop_s: break

        skip = 1

        pitches = array(pitches[skip:])
        confidences = array(confidences[skip:])
        times = [t * self.hop_s for t in range(len(pitches))]
        #import pdb;pdb.set_trace()
        cleaned_pitches = pitches
        #cleaned_pitches = ma.masked_where(cleaned_pitches < 0, cleaned_pitches)
        #cleaned_pitches = ma.masked_where(cleaned_pitches > 120, cleaned_pitches)
        #cleaned_pitches = ma.masked_where(confidences < tolerance, cleaned_pitches)
        #cleaned_pitches = [ i for i in cleaned_pitches if i !=0 ]
        return cleaned_pitches
    
    def compress_seq(self, seq, **kwargs):
        k_size = 4
        #import pdb; pdb.set_trace()
        if 'kernel_size' in kwargs:
            k_size = kwargs['k_size']
        seq = seq.tolist()
        # split in 4-items and calc median 
        out_seq = stats.mode(np.reshape(np.pad(seq, (0, 4-(np.mod(len(seq),4))),mode='constant'), (-1, 4)),1)[0].squeeze()
        
        return out_seq    
            


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='input audio file')
    parser.add_argument('-o', '--output', type=str, help='store seq file')
    parser.add_argument('--bpm', type=int, help='beats per minute')
    parser.add_argument('--tpb', type=int, help='ticks per beat')
    parser.add_argument('--type', type=str, help='type of representation')
    parser.add_argument('--hparams', type=str,
                        required=False, help='comma separated name=value pairs')

    args = parser.parse_args()
    #hparams = create_hparams(args.hparams)
    sequencer = Sequencer(bpm=args.bpm, tpb=args.tpb)
    seq = sequencer.to_seq(args.input)
    if args.type=='compress4':
        seq = sequencer.compress_seq(seq)  
    if len(seq) == 0:
        print(f'Warining: file {args.input} has no sequence')
        sys.exit(1)
    seq_str=[]
    
    # normalize
    zero_count=0
    for pitch in seq:
        if pitch == 0:
            pitch = 1
            zero_count+=1
        else:
            if pitch > 95:
                pitch = 95
            if pitch < 37:
                pitch = 37
            pitch = pitch - 35
        seq_str.append(str(pitch))
    #print (len(seq))
    if zero_count > (len(seq) * 0.8) :
        print('found %s zeros from %s' % (zero_count, len(seq)))
    else:
        seqs = ','.join(seq_str)
        if args.output:
            with open(args.output, 'w') as fh:
                print( '[' + seqs + ']', file=fh)
        else:
            print (seq)
      
