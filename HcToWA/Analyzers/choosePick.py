import os
import argparse
import numpy as np
from ROOT import TFile, TH1D

# argparse
parser = argparse.ArgumentParser()
parser.add_argument("--sig", default=None, required=True, type=str, help="Signal mass point")
parser.add_argument("--mean", default=None, required=True, type=float, help="mean of mass pick")
parser.add_argument("--sigma", default=None, required=True, type=float, help="sigma of mass pick")
args = parser.parse_args()

if __name__=="__main__":
    # for signal sample
    # choose 2sigma varation of dimuon mass
    # store it to Nsig with weight
    # for background sample
    # do the same
    # calculate Nsig/sqrt(Nbkg)
    bkgs = ['DYJets10to50_MG', 'DYJets', 'TTLL_powheg',         # fakes
            'ttHToNonbb', 'ttWToLNu', 'ttZToLLNuNu', 'TTG',     # ttX
            'WZTo3LNu_powheg', 'ZGToLLG_01J', 'ZZTo4L_powheg',  # VV
            'ggHToZZTo4L', 'VBF_HToZZTo4L',                     # HtoZZto4l
            'WWW', 'WWZ', 'WZZ', 'ZZZ']                         # VVV
    f_sig = TFile("Samples/Preselection_TTToHcToWA_AToMuMu_"+args.sig+".root")
    mean = args.mean      # GeV
    sigma = args.sigma    # GeV
    f_bkgs = []
    for bkg in bkgs:
        f_bkgs.append(TFile("Samples/Preselection_"+bkg+".root"))

    Nsig = 0.
    Nbkg = 0.
    Ntotal_sig = 0.
    Ntotal_bkg = 0.
    for event in f_sig.Events:
        mMuMu = event.mMuMu
        weight = event.weight
        Ntotal_sig += weight
        if mean-3*sigma < mMuMu and mMuMu < mean+3*sigma:
            Nsig += weight
        else:
            continue
    for f in f_bkgs:
        for event in f.Events:
            mMuMu = event.mMuMu
            weight = event.weight
            Ntotal_bkg += weight
            if mean-3*sigma < mMuMu and mMuMu < mean+3*sigma:
                Nbkg += weight
            else:
                continue
    print(args.sig)
    print("Signal eff:", Nsig/Ntotal_sig)
    print("Background rejection:", 1.-(Nbkg/Ntotal_bkg))
    print("Nsig/sqrt(Nbkg):", Nsig/np.sqrt(Nbkg))
    print()

    


