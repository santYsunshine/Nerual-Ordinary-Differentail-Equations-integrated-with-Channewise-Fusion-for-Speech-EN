import pyloudnorm as pyln
import soundfile as sf
import glob
import argparse
import os

def main(args):
    files = glob.glob(args.inDir + '/*.wav')
    _, sample_rate = sf.read(files[0])
    ref_sound_level = -18
    loudness_meter = pyln.Meter(sample_rate)
    for f in files:
        audio, sr = sf.read(f)
        assert sr == sample_rate, 'sample rate not equal to first file'
        audio_level = loudness_meter.integrated_loudness(audio)
        gain_db = ref_sound_level - audio_level
        g = 10 ** (gain_db / 20.)
        audio = audio * g
        filename = f.split('/')[-1].split('.')[0] + '_aligned.wav'
        sf.write(os.path.join(args.indir, filename), audio, sample_rate)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("noisy data generator")
    #parser.add_argument('--inDir', type=str, default='normal',
    #                    help='select noise dataset: Urban, Urbantest, dir_weighted(include folders to be weighted in noiseDir), normal')
    parser.add_argument('--inDir', type=str, default='/media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/tt/spk2',
                        help='select noise dataset: Urban, Urbantest, dir_weighted(include folders to be weighted in noiseDir), normal')
    
    args = parser.parse_args()
    print(args)
    main(args)