import argparse
import json
import pickle
import numpy as np
from data_utils.data_handler import clearup
from keras.models import load_model
import pandas as pd
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', '-dr', default="data/features_full", type=str,
                        help='Provide the path for PathReports')
    parser.add_argument('--metadata_file', default="data/histo_metadata.csv", type=str,
                        help='Metadata file')
    parser.add_argument('--model_h5', default="mt_cnn_model.h5", type=str,
                        help='Model')
    parser.add_argument('--word2idx', default="data/word2idx.pkl", type=str,
                        help='word2idx')
    parser.add_argument('--vocab', default="data/vocab.npy", type=str,
                        help='Vocab')
    parser.add_argument('--histology_mapper', default="data/mapper/histology_class_mapper.json", type=str,
                        help='Histology class mapper')
    parser.add_argument('--site_mapper', default="data/mapper/site_class_mapper.json", type=str,
                        help='Site class mapper')

    parser.add_argument('--filename', 
        default="TCGA-2A-AAYO.3889AA76-F350-4DA4-987B-79E8D2349262.txt", 
        type=str,
        help = "Path to pathology report to use in inference")
    args = parser.parse_args()
    print(args)
    return args



def vectorSingle(filename,word2idx,vocab, args):
    fname = args.data_dir + "/" + filename.split('.hstlgy')[0].strip()
    doc = open(fname, 'r', encoding="utf8").read().strip()
    doc = clearup(doc)

    max_len = 1500
    unk = len(vocab)
    # convert words to indices
    text_idx = np.zeros((1, max_len))
    #for i, sent in enumerate(doc):
    singleDocVec = [word2idx[word] if word in word2idx else unk for word in doc][:max_len]
    l = len(singleDocVec)
    text_idx[0, :l] = singleDocVec
    return text_idx
def originalLabels(filename):
    print("Original Labels")
    labelList=pd.read_csv(args.metadata_file,delimiter='\t')
    rows=labelList.loc[labelList['filename'].str.contains(filename)]
    print(rows)
    return

def mtcnnPredict(filename,word2idx,vocab,siteIdtoLabel,histologyIdtoLabel, args):
    vec = vectorSingle(filename, word2idx, vocab, args)
    model = load_model(args.model_h5)
    y_preds = model.predict(vec)
    sitePred = np.argmax(y_preds[0], 1)
    histoPred = np.argmax(y_preds[1], 1)

    print("MTCNN Prediction")
    print(siteIdtoLabel[sitePred[0]])
    print(histologyIdtoLabel[histoPred[0]])
    return
if __name__ == '__main__':

    args = parse_arguments()
    with open(args.word2idx, 'rb') as f:
        word2idx = pickle.load(f)

    vocab = np.load(args.vocab)
    filename=args.filename

    with open(args.histology_mapper) as json_file:
        histologyLabel = json.load(json_file)
        histologyIdtoLabel = {}
        for k, v in histologyLabel.items():
            histologyIdtoLabel[v] = k

    with open(args.site_mapper) as json_file:
        siteLabel = json.load(json_file)
        siteIdtoLabel = {}
        for k, v in siteLabel.items():
            siteIdtoLabel[v] = k


    # Prediction using MTCNN
    mtcnnPredict(filename,word2idx,vocab,siteIdtoLabel,histologyIdtoLabel, args)

    # Original Labels
    originalLabels(filename)

