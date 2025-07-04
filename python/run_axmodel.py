import os
import glob
import argparse
import numpy as np
import torch
import torchaudio
from processor import FBank

import axengine as axe
from axengine import axclrt_provider_name, axengine_provider_name

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="res2netv2.axmodel", help="axmodel path")
parser.add_argument('--wavs', nargs='+', type=str, help='Wavs')
parser.add_argument("--samplerate", type=int, default=16000, help="Specify the audio sample rate in Hz (default is 16,000)")
parser.add_argument("--max_frames", type=int, default=360, help="the max audio frames")

def load_wav(wav_file, obj_fs=16000):
    """Read the audio from the specified source path. """
    wav, fs = torchaudio.load(wav_file)
    if fs != obj_fs:
        print(f'[WARNING]: The sample rate of {wav_file} is not {obj_fs}, resample it.')
        wav, fs = torchaudio.sox_effects.apply_effects_tensor(
            wav, fs, effects=[['rate', str(obj_fs)]]
        )
    if wav.shape[0] > 1:
        wav = wav[0, :].unsqueeze(0)
    return wav
    
def axmodel_inference(onnx_path, features, cuda=True):
    """Perform inference using an axmodel. """
    def from_numpy(x):
        return x if isinstance(x, np.ndarray) else np.array(x)
    
    session = axe.InferenceSession(onnx_path, providers='AxEngineExecutionProvider')
    output_names = [x.name for x in session.get_outputs()]
    input_name = session.get_inputs()[0].name

    # onnx inference
    y = session.run(output_names, {input_name: features})

    if isinstance(y, (list, tuple)):
        y = from_numpy(y[0]) if len(y) == 1 else [from_numpy(x) for x in y]
    else:
        y = from_numpy(y)
    return y
        
def compute_embedding(wav_file, model, frames=360, obj_fs=16000):
    # load wav
    wav = load_wav(wav_file, obj_fs)

    # compute feat
    feature_extractor = FBank(80, obj_fs, mean_nor=True)
    feat = feature_extractor(wav).unsqueeze(0)

    # Adjust the shape of feature to [1, 1, frames, 80]
    shape = list(feat.shape)
    if shape[1] >= frames:
        feat = feat.narrow(1, 0, frames)
    else:
        shape[1] = frames
        feat = feat.new_full(shape, fill_value=0)
    
    feat = feat.permute(1, 2, 0).unsqueeze(0).cpu().numpy()

    # compute embedding
    embedding = axmodel_inference(model, feat).squeeze(0)
    
    return embedding

def main():
    args = parser.parse_args()
    if args.wavs is None or len(args.wavs) == 2:
        if args.wavs is None:
            try:
                # use example wavs
                examples_dir = './wav'
                wav_path1, wav_path2 = list(glob.glob(os.path.join(examples_dir, '*.wav')))[0:2]
                print(f'[INFO]: No wavs input, use example wavs instead.')
            except:
                assert Exception('Invalid input wav.')
        else:
            # use input wavs
            wav_path1, wav_path2 = args.wavs
        
        embedding1 = compute_embedding(wav_path1, args.model)
        embedding2 = compute_embedding(wav_path2, args.model)

        # compute similarity score
        print('[INFO]: Computing the similarity score...')
        similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
        scores = similarity(torch.from_numpy(embedding1).unsqueeze(0), torch.from_numpy(embedding2).unsqueeze(0)).item()
        print('[INFO]: The similarity score between two input wavs is %.4f' % scores)
    
if __name__ == '__main__':
    main()
    