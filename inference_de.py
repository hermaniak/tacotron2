import argparse

import os
import pandas as pd

import sys
sys.path.append('waveglow/')
import numpy as np
import torch

from hparams import create_hparams
from train import load_model
from text import text_to_sequence
from denoiser import Denoiser

import scipy

import time

from utils import load_wav_to_torch, load_filepaths_and_text

def get_pitch(text):
        pitch = list(map(int, text.replace('[','').replace(']','').split(',')))
        #pitch = [ (i - 36) for i in pitch if i > 37 and i < 95 ]
        return pitch



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='PyTorch Tacotron 2 Inference Server')

    parser.add_argument('-i', '--input_file', type=str, required=True,
                        help='full path to the input texts (id, text)')
    parser.add_argument('-o', '--output_dir', required=True,
                        help='output folder to save audio (file per text)')
    parser.add_argument('--suffix', type=str, default="", help="output filename suffix")
    parser.add_argument('--tacotron2', type=str,
                        help='full path to the Tacotron2 model checkpoint file',)
    parser.add_argument('--waveglow', type=str,
                        help='full path to the WaveGlow model checkpoint file')
    parser.add_argument('-sr', '--sampling-rate', default=22050, type=int,
                        help='Sampling rate')
    args, _ = parser.parse_known_args()

    input_file = args.input_file
    audiopaths_and_seq = load_filepaths_and_text(input_file)
    print(f'The total number of sentences to process is {len(audiopaths_and_seq)}')
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_suffix = args.suffix
    tacotron2_path = args.tacotron2
    waveglow_path = args.waveglow
    sampling_rate = args.sampling_rate

    setup_start = time.time()

    hparams = create_hparams()
    hparams.sampling_rate = sampling_rate

    model = load_model(hparams)
    model.load_state_dict(torch.load(tacotron2_path)['state_dict'])
    _ = model.cuda().eval().half()
    waveglow = torch.load(waveglow_path)['model']
    waveglow.cuda().eval().half()
    for k in waveglow.convinv:
        k.float()
    denoiser = Denoiser(waveglow)
    #import pdb;pdb.set_trace() 
    setup_end = time.time()
    setup_time = setup_end - setup_start
    print(f'The setup time is {setup_time} seconds')

    processing_start = time.time()

    for wav, seq in audiopaths_and_seq:
        sequence = np.array(get_pitch(seq))[None, :]

        #import pdb;pdb.set_trace() 
        # Prepare the text
        #sequence_text = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
        #sequence = np.array(audio_to_seq(text))[None, :]
        
        sequence = torch.autograd.Variable(
            torch.from_numpy(sequence)).cuda().long()
        print(sequence)
        # Decode the text to the mel-spectrogram
        mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
        # Convert the mel-spectrogram to audio
        with torch.no_grad():
            audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)
        audio_denoised = denoiser(audio, strength=0.01)[:, 0]
        scipy.io.wavfile.write(f'{output_dir}/{output_suffix}_{os.path.basename(wav)}', hparams.sampling_rate, audio_denoised[0].cpu().numpy())

    processing_end = time.time()
    processing_time = processing_end - processing_start
    print(f'Processed {len(audiopaths_and_seq)} sentences in {processing_time} seconds')
    processing_time_per_sentence = round(processing_time / len(audiopaths_and_seq), 1)
    print(f'Each sentence took {processing_time_per_sentence} seconds')

