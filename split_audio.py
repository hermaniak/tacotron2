from audio import audio_to_seq
from aubio import source, sink
import logging
import os, sys, argparse 
import numpy as np
#from hparams import create_hparams


min_time = 1200
max_time = 5000
#hop_s = 1024 
hop_s = 256
win_s = 1024
#win_s = 4096

logging.basicConfig(level=logging.INFO)

class Kernel():
    def __init__(self, out, listname):
        self.len = 0
        self.pause_len = 0
        self.start = 0 
        self.seq=[]
        self.filelist= listname
        self.out_prefix = out
        self.min_seq = min_time * 22050 / (hop_s * 1000) 
        self.max_seq = max_time * 22050 / (hop_s * 1000)  
        self.out_file = '%s_sample_0.wav' % (out)
        self.snk = sink(self.out_file,  samplerate=22050, channels=1)
        logging.info('kernel: min=%s max=%s' % (self.min_seq, self.max_seq))

    def reset(self, i):
        self.len = 0
        self.pause_len = 0
        self.start = 0
        self.seq=[]
        self.snk.close()  
        self.out_file='%s_sample_%s.wav' % (self.out_prefix, i)      
        self.snk = sink(self.out_file, samplerate=22050, channels=1)

    def write(self, samples):
        #import pdb;pdb.set_trace()
        self.snk(samples.astype(np.float32), len(samples))

    def write_seq(self):
        logging.info('split %s with len %s' % (self.out_file, self.len))
        with open(self.filelist, 'a') as fh:
            print ('%s|%s' % ( self.out_file, self.seq), file=fh)

#def slice_audio_fix(audio_file, out_prefix, filelist_name, **kwargs):
#    #import pdb;pdb.set_trace()
#    if 'bpm' in kwargs and 'beats' in kwargs:
#        split_time = 60000 / float(kwargs['pbm'] * kwargs['beats'] 
#    elif 'split_time' in kwargs:
#        split_time = kwargs['split_time']
#    split_samples = split_time * 22050 / 1000
#    
#    
#    seq = audio_to_seq(audio_file, **kwargs)
#    samplerate=22050
#    split_hops = split_samples
#    s = source(audio_file, samplerate, split_samples)
#    i=0
#    cnts={}
#    #print(seq)
 

def slice_audio(audio_file, out_prefix, filelist_name, **kwargs):
    #import pdb;pdb.set_trace()
    seq = audio_to_seq(audio_file, **kwargs)
    k=Kernel(out_prefix, filelist_name)
    samplerate=22050
    s = source(audio_file, samplerate, kwargs['hop_s'])
    i=0
    cnts={}
    #print(seq)
    for pitch in seq:
        logging.debug('p: %s - idx: %s - kernel(len: %s, plen: %s)' % ( pitch, i, k.len, k.pause_len ))
        samples, read = s()
        if pitch == 0:
            if k.len == 0:
                continue

            if k.len > k.min_seq and k.pause_len > 1:
                i += 1
                k.write_seq()
                k.reset(i)
                continue
            #else:
            #    print(f'len {k.len} too short')
            if k.len > 2:
                # intermediate pause, ok
                k.seq.append(1)
                k.len += 1
                k.write(samples)
    
            if k.pause_len > 10:
                i += 1 
                k.reset(i)
                logging.info('to short segment, reset')
            k.pause_len += 1 
            

        else:
            if pitch not in cnts:
                cnts[pitch] = 1
            else:
                cnts[pitch] += 1
            if k.len > k.max_seq:
                logging.info('reached max, flush')
                i += 1
                k.write_seq()
                k.reset(i)

            k.write(samples)
            if pitch > 95:
               pitch = 95
            if pitch < 37:
                pitch = 37
            pitch = pitch - 35 

            k.seq.append(pitch)
            k.len += 1
            k.pause_len = 0
                 
    #print('Statistics:')
    #print('   unique pitch count: %s' % len(cnts.keys()))
    #print('   pitches:')
    #for k, v in cnts.items():
    #    print('     %s - %s' % (k,v))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='input audio file')
    parser.add_argument('-o', '--output', type=str, help='output file prefix. Files will be stored to prefix_samplexxx.wav')
    parser.add_argument('-l', '--list', type=str, default='filelist.txt', help='output filelist')
    parser.add_argument('-s', '--seq_len', type=float, default='0', help='sequent length')
    parser.add_argument('--hparams', type=str,
                        required=False, help='comma separated name=value pairs')

    args = parser.parse_args()
    #hparams = create_hparams(args.hparams)
    #print (hparams)
    #hop_length=256,win_length=1024
    slice_audio(args.input, args.output, args.list, hop_s=hop_s, win_s=win_s, seq_len=args.seq_len)

