import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from ROOT import TFile, TH1D, TH2D
from tensorflow.keras import Model, Input, layers, optimizers, callbacks
from tensorflow.keras.utils import to_categorical

# argparse
parser = argparse.ArgumentParser(description="train mode or test mode")
parser.add_argument("--train", default=False, action='store_true', help='activate training mode')
parser.add_argument("--sig", default=None, required=True, type=str, help="Signal mass point")
args = parser.parse_args()

def create_model(n_features, n_classes, dropout=0.5):
    x = Input(shape=n_features)
    y = layers.Dense(128, activation='relu', kernel_regularizer='l2')(x)
    y = layers.Dropout(dropout)(y)
    y = layers.Dense(128, activation='relu', kernel_regularizer='l2')(y)
    y = layers.Dropout(dropout)(y)
    y = layers.Dense(128, activation='relu', kernel_regularizer='l2')(y)
    y = layers.Dropout(dropout)(y)
    y = layers.Dense(n_classes, activation='softmax')(y)
    model = Model(x, y)
    optimizer = optimizers.Adam(learning_rate=0.001, amsgrad=True)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def preprocess(processes, training=False, normalize=False, split=1.0):
    temp_inputs = []
    temp_labels = []
    temp_weights = []
    label = -1
    max_events = 0  # upper bound to signal events
    for process in processes:
        label += 1
        f = TFile.Open("Samples/Preselection_" + process + ".root")
        process_inputs = []
        process_labels = []
        process_weights = []
        for event in f.Events:
            input_mass = [event.mMuMu, event.mMuMuJJ]
            input_leptons = [event.ptMu1, event.etaMu1, event.phiMu1, event.ptMu2, event.etaMu2, event.phiMu2, event.ptEle, event.etaEle, event.phiEle]
            input_jets = [event.ptJ1, event.etaJ1, event.phiJ1, event.ptJ2, event.etaJ2, event.phiJ2, event.ptB1, event.etaB1, event.phiB1, event.Nj, event.Nb]
            input_deltaR = [event.dRl1l2, event.dRl1l3, event.dRl2l3, event.dRj1l1, event.dRj1l2, event.dRj1l3, event.dRj2l1, event.dRj2l2, event.dRj2l3, event.dRb1l1, event.dRb1l2, event.dRb1l3, event.dRj1j2]
            input_T = [event.HT, event.LT, event.MET, event.ST, event.HToverST, event.LToverST]
            temp_input = input_mass + input_leptons + input_jets + input_deltaR + input_T
            #temp_input = input_leptons + input_jets

            process_inputs.append(temp_input)
            process_labels.append(label)
            process_weights.append(event.weight)
            if label == 0:
                max_events += 1
        # for training, balance nevents
        if training:
            if len(process_inputs) > max_events:
                process_inputs = process_inputs[:max_events]
                process_labels = process_labels[:max_events]
                process_weights = process_weights[:max_events]
        temp_inputs += process_inputs
        temp_labels += process_labels
        temp_weights += process_weights

    # for training, shuffle all processes
    inputs = []
    labels = []
    weights = []
    if training:
        shuf_idx = np.arange(len(temp_inputs))
        np.random.shuffle(shuf_idx)
        for idx in shuf_idx:
            inputs.append(temp_inputs[idx])
            labels.append(temp_labels[idx])
            weights.append(temp_weights[idx])
    # no need to shuffle for evaluation mode
    else:
        inputs = temp_inputs
        labels = temp_labels
        weights = temp_weights
    
    # standarization
    inputs = np.array(inputs)
    labels = to_categorical(np.array(labels))
    weights = np.array(weights)
    if normalize:
        inputs = (inputs - np.mean(inputs, axis=0)) / np.std(inputs, axis=0)
    if training:
        ratio = int(labels.shape[0]*split)
        train_inputs, test_inputs = inputs[:ratio], inputs[ratio:]
        train_labels, test_labels = labels[:ratio], labels[ratio:]
        train_weights, test_weights = weights[:ratio], weights[ratio:]
        df_inputs = {}; df_labels = {}; df_weights = {};
        df_inputs["train"] = train_inputs
        df_inputs["test"] = test_inputs
        df_labels["train"] = train_labels
        df_labels["test"] = test_labels
        df_weights["train"] = train_weights
        df_weights["test"] = test_weights
        return df_inputs, df_labels, df_weights
    else:
        return inputs, labels, weights

def plotter(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric])
    plt.title('Model '+metric)
    plt.xlabel('Epoch')
    plt.ylabel(metric)
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('Results/'+args.sig+'_'+metric+'.png')

if __name__=="__main__":
    EPOCHS = 2000
    BATCH_SIZE = 8192
    VAL_SPLIT = 0.25
    checkpoint_path = "Models/SimpleDNN_" + args.sig + ".ckpt"

    # data preprocessing
    processes_train = ['TTToHcToWA_AToMuMu_'+args.sig, 'ttHToNonbb', 'ttWToLNu', 'ttZToLLNuNu']
    processes_valid = ['TTToHcToWA_AToMuMu_'+args.sig,
            'ttHToNonbb', 'ttWToLNu', 'ttZToLLNuNu', 'TTG',         # ttX
            'DYJets', 'DYJets10to50_MG', 'TTLL_powheg',             # fakes
            'WZTo3LNu_powheg', 'ZGToLLG_01J', 'ZZTo4L_powheg',      # VV
            'ggHToZZTo4L', 'VBF_HToZZTo4L',                         # HtoZZto4l
            'WWW', 'WWZ', 'WZZ', 'ZZZ']                              # VVV 
    if args.train:
        inputs, labels, weights = preprocess(processes_train, args.train, normalize=True, split=0.9)
        train_inputs, test_inputs = inputs['train'], inputs['test']
        train_labels, test_labels = labels['train'], labels['test']
        train_weights, test_weights = labels['train'], labels['test']
   
        n_features = train_inputs.shape[1]
        model = create_model(n_features, len(processes_train))
        callbacks = [
            callbacks.EarlyStopping(patience=15, monitor='val_loss'), 
            callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=True)
            ]
        history = model.fit(train_inputs, train_labels, 
            validation_split=VAL_SPLIT, epochs=EPOCHS, batch_size=BATCH_SIZE,
            callbacks=callbacks)

        # monitor training
        plotter(history, 'accuracy')
        plotter(history, 'loss')
        
        test_loss, test_acc = model.evaluate(test_inputs, test_labels)
    else:
        n_classes = len(processes_valid)
        # data is not shuffled for validation mode
        inputs, labels, weights = preprocess(processes_valid, args.train, normalize=False)
        # normalize inputs before feeding to the model
        normed_inputs = (inputs - np.mean(inputs, axis=0)) / np.std(inputs, axis=0)
        # maximum label is 3
        restricted_labels = []
        for i in range(len(labels)):
            this_label = np.argmax(labels[i])
            if this_label > 3:
                restricted_labels.append(np.array([0., 0., 0., 1.]))
            else:
                temp = np.zeros(4)
                temp[this_label] = 1.
                restricted_labels.append(temp)
        restricted_labels = np.array(restricted_labels)

        n_features = inputs.shape[1]
        model = create_model(n_features, len(processes_train))
        loss, acc = model.evaluate(normed_inputs, restricted_labels, verbose=2)
        print("훈련되지 않은 모델의 정확도: {:5.2f}%".format(100*acc))
        # restore weights
        model.load_weights(checkpoint_path)
        loss, acc = model.evaluate(normed_inputs, restricted_labels, verbose=2)
        print("복원된 모델의 정확도: {:5.2f}%".format(100*acc))

        prediction = model.predict(normed_inputs)
        # Draw confusion matrix
        f = TFile("Results/SimpleDNN_"+args.sig+".root", "recreate")
        hist_mMuMu = TH1D("mMuMu", "", 200, 0., 200.)
        conf_matrix = TH2D("confusion_matrix", "", len(processes_train), 0, len(processes_train),
                                                   len(processes_valid), 0, len(processes_valid))

        lst_classes = np.zeros(n_classes)
        for i in range(len(inputs)):
            lst_classes[np.argmax(labels[i])] += 1
        n_sig = 0.
        n_total_sig = 0.
        n_bkg = 0.
        n_total_bkg = 0.
        for i in range(len(inputs)):
            conf_matrix.Fill(np.argmax(prediction[i]), np.argmax(labels[i]), 
                    1/lst_classes[np.argmax(labels[i])])
            mMuMu = inputs[i][0] # mass at the first idx
            label = np.argmax(labels[i])
            predict = np.argmax(prediction[i])
            weight = weights[i]
            if label == 0:
                n_total_sig += weight
                if predict == 0:
                    n_sig += weight
                    hist_mMuMu.Fill(mMuMu, weight)
            else:
                n_total_bkg += weight
                if predict == 0:
                    n_bkg += weight
                    hist_mMuMu.Fill(mMuMu, weight)
        print("Nsig/Sqrt(Nbkg):", n_sig/np.sqrt(n_bkg))
        print("Signal eff.:", n_sig/n_total_sig)
        print("Bkg rej.:", 1.-n_bkg/n_total_bkg)
        f.cd()
        conf_matrix.SetStats(0)
        hist_mMuMu.SetStats(0)
        conf_matrix.Write()
        hist_mMuMu.Write()
        f.Close()
