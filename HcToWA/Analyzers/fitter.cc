#include "RooRealVar.h"
#include "RooDataSet.h"
#include "RooDataHist.h"
#include "RooGaussian.h"
#include "RooPlot.h"
using namespace RooFit;

void fitter() {
	// mass points
	// 70 - 15 40 65
	// 100 - 15 25 60 95
	// 130 - 15 45 55 90 125
	// 160 - 15 45 75 85 120 155
	TFile* f = new TFile("Samples/Preselection_TTToHcToWA_AToMuMu_MHc160_MA45.root");
	TH1D* h = (TH1D*)f->Get("Pre_1e2mu/mMuMu_finebinning");
	h->Rebin(20);

	// Prepare dataset
	double hmin = h->GetXaxis()->GetXmin();
	double hmax = h->GetXaxis()->GetXmax();
	RooRealVar x("x", "x", hmin, hmax);
	RooDataHist dh("dh", "dh", x, Import(*h));

	RooPlot* frame = x.frame(Title("dimuon mass peak"));
	dh.plotOn(frame, MarkerColor(2), MarkerSize(0.9), MarkerStyle(21));
	dh.statOn(frame);

	// Prepare pdf
	RooRealVar mean("mean", "mean", 35., 45., 55.);
	RooRealVar width("width", "width", 2., 0., 4.);
	RooRealVar sigma("sigma", "sigma", 1., 0., 2.);
	//RooGaussian gauss("gauss", "gauss", x, mean, sigma);
	RooVoigtian gauss("guass", "gauss", x, mean, width, sigma);
	
    RooFitResult* filters = gauss.fitTo(dh, "qr");
	gauss.plotOn(frame, LineColor(4));

	TCanvas* cvs = new TCanvas("cvs", "", 800, 600);
	cvs->cd(); gPad->SetLeftMargin(0.15);
	frame->Draw();
}
