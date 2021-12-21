#!/usr/bin/python3
# -*-coding: utf-8 -
'''
    @author:  MD. Nazmuddoha Ansary
'''
#--------------------
# imports
#--------------------
import sys
sys.path.append('../')

import argparse
import os 
import json
import pandas as pd 

from tqdm import tqdm
from ast import literal_eval

from coreLib.utils import *
from coreLib.processing import processData
from coreLib.store import createRecords
from coreLib.vocab import vocabs
tqdm.pandas()
#--------------------
# main
#--------------------
def main(args):
    config_json  =   "../config.json"
    data_dir    =   args.data_dir
    img_height  =   int(args.img_height)
    img_width   =   int(args.img_width)
    iden        =   args.iden
    seq_max_len =   int(args.seq_max_len)
    tf_size     =   int(args.tf_size)
    vocab_iden  =   args.vocab_iden

    assert vocab_iden in vocabs.keys(),"invalid vocab name"
    assert iden is not None,"iden not found"
    
    vocab=vocabs[vocab_iden]
    img_dim=(img_height,img_width)
    csv=os.path.join(data_dir,"data.csv")
    # processing
    df=processData(csv,vocab,seq_max_len,img_dim)
    # storing
    save_path=create_dir(data_dir,iden)
    LOG_INFO(save_path)
    createRecords(df,save_path,tf_size)

    config={"vocab":vocab,
            "pos_max":seq_max_len,
            "img_height":img_height,
            "img_width" :img_width,
            "tf_size":tf_size,
            "vocab_iden":vocab_iden,
            "zip_iden":iden}

    with open(config_json, 'w') as fp:
        json.dump(config, fp,sort_keys=True, indent=4,ensure_ascii=False)
#-----------------------------------------------------------------------------------

if __name__=="__main__":
    '''
        parsing and execution
    '''
    parser = argparse.ArgumentParser("Recognizer Dataset Creating Script")
    parser.add_argument("data_dir", help="Path of the source data folder that contains langauge datasets")
    parser.add_argument("iden",help="identifier to identify the dataset")
    parser.add_argument("--seq_max_len",help=" the maximum length of data for modeling")
    parser.add_argument("--vocab_iden",help=" the vocabulary to use. available: english_numbers,bangla_numbers,english_all,bangla_all,all")
    parser.add_argument("--tf_size",required=False,default=128,help=" the size of  data to store in 1 tfrecord:default=128")
    parser.add_argument("--img_height",required=False,default=64,help ="height for each grapheme: default=64")
    parser.add_argument("--img_width",required=False,default=512,help ="width for each grapheme: default=512")
    args = parser.parse_args()
    main(args)