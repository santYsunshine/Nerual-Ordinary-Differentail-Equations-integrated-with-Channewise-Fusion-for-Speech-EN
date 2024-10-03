#!/usr/bin/env python

# wujian@2018

import os
import argparse
import time

import torch as th
import numpy as np

from main.nnet.model import ConvTasNet

from libs.utils import load_json, get_logger
from libs.audio import WaveReader, write_wav

logger = get_logger(__name__)


class NnetComputer(object):
    def __init__(self, cpt_dir, gpuid, cpt_epoch):
        self.device = th.device(
            "cuda:{}".format(gpuid)) if gpuid >= 0 else th.device("cpu")
        nnet = self._load_nnet(cpt_dir, cpt_epoch)
        self.nnet = nnet.to(self.device) if gpuid >= 0 else nnet
        # set eval model
        self.nnet.eval()

    def _load_nnet(self, cpt_dir, cpt_epoch):
        nnet_conf = load_json(cpt_dir, "mdl.json")
        nnet = ConvTasNet(**nnet_conf)
        cpt_fname = os.path.join(cpt_dir, "{}.pt.tar".format(cpt_epoch))
        cpt = th.load(cpt_fname, map_location="cpu")
        nnet.load_state_dict(cpt["model_state_dict"])
        logger.info("Load checkpoint from {}, epoch {:d}".format(
            cpt_fname, cpt["epoch"]))
        return nnet

    def compute(self, samps):
        with th.no_grad():
            raw = th.tensor(samps, dtype=th.float32, device=self.device)
            start = time.time()
            sps = self.nnet(raw)
            end = time.time()
            print('duration: ', end-start)
            sp_samps = [np.squeeze(s.detach().cpu().numpy()) for s in sps]
            return sp_samps


def run(args):
    mix_input = WaveReader(args.input, sample_rate=args.fs)
    computer = NnetComputer(args.checkpoint, args.gpu, args.cpt_epoch)
    for key, mix_samps in mix_input:
        logger.info("Compute on utterance {}...".format(key))
        gpuRamMax = 16000*args.RamTime
        print('mix_samps.size: ', mix_samps.size)
        tstart = time.time()
        if mix_samps.size > gpuRamMax:
            splitNum = mix_samps.size//gpuRamMax + 1
            # spks = [np.array(), np.array()]
            # spksAll = []
            spks = np.array([])
            spksAll = []
            for i in range(splitNum):
                mixSample = mix_samps[i*gpuRamMax:(i+1)*gpuRamMax]
                spks_i = computer.compute(mixSample)
                spksNpArray = np.array(spks_i)
                spksAll.append(spksNpArray)
                # spks = np.concatenate((spks,spksNpArray),axis=1) if spks.size else spksNpArray
            spks_tuples = tuple(spksAll)
            spks = np.concatenate(spks_tuples,axis=1)
            spks = list(spks)
        else:
            spks = computer.compute(mix_samps)
        tend = time.time()
        print('denoise time: ', tend - tstart)
        # print('mix_samps type: ', type(mix_samps))
        # print('spks type: ', type(spks))
        norm = np.linalg.norm(mix_samps, np.inf)
        for idx, samps in enumerate(spks):
            samps = samps[:mix_samps.size]
            # norm
            samps = samps * norm / np.max(np.abs(samps))
            write_wav(
                os.path.join(args.dump_dir, "spk{}/{}.wav".format(
                    idx + 1, key)),
                samps,
                fs=args.fs)
    logger.info("Compute over {:d} utterances".format(len(mix_input)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Command to do speech separation in time domain using ConvTasNet",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("checkpoint", type=str, help="Directory of checkpoint")
    parser.add_argument("--cpt_epoch", type=str, default='best', help="epoch of checkpoint")
    parser.add_argument(
        "--input", type=str, required=True, help="Script for input waveform")
    parser.add_argument(
        "--gpu",
        type=int,
        default=-1,
        help="GPU device to offload model to, -1 means running on CPU")
    parser.add_argument(
        "--fs", type=int, default=8000, help="Sample rate for mixture input")
    parser.add_argument(
        "--dump-dir",
        type=str,
        default="sps_tas",
        help="Directory to dump separated results out")
    parser.add_argument(
        "--RamTime", type=int, default=300, help="max audio length in sec")
    args = parser.parse_args()
    run(args)

