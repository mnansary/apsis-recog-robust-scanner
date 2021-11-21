# -*-coding: utf-8 -
'''
    @author:  MD. Nazmuddoha Ansary
'''
#--------------------
# imports
#--------------------
import random
import pandas as pd 
import cv2
import math
from tqdm import tqdm
from .utils import *
tqdm.pandas()
#--------------------
# helpers
#--------------------
not_found=[]
def pad_label(x,max_len,pad_value,start_end_value):
    '''
        lambda function to create padded label for robust scanner
    '''
    if len(x)>max_len-2:
        return None
    else:
        if start_end_value is not None:
            x=[start_end_value]+x+[start_end_value]
        pad=[pad_value for _ in range(max_len-len(x))]
        return x+pad
        
def encode_label(x,vocab):
    '''
        encodes a label
    '''
    global not_found
    label=[]
    for ch in x:
        try:
            label.append(vocab.index(ch))
        except Exception as e:
            if ch not in not_found:not_found.append(ch)
    return label

def padWordImage(img,pad_loc,pad_dim,pad_type,pad_val):
    '''
        pads an image with white value
        args:
            img     :       the image to pad
            pad_loc :       (lr/tb) lr: left-right pad , tb=top_bottom pad
            pad_dim :       the dimension to pad upto
            pad_type:       central or left aligned pad
            pad_val :       the value to pad 
    '''
    
    if pad_loc=="lr":
        # shape
        h,w,d=img.shape
        if pad_type=="central":
            # pad widths
            left_pad_width =(pad_dim-w)//2
            # print(left_pad_width)
            right_pad_width=pad_dim-w-left_pad_width
            # pads
            left_pad =np.ones((h,left_pad_width,3))*pad_val
            right_pad=np.ones((h,right_pad_width,3))*pad_val
            # pad
            img =np.concatenate([left_pad,img,right_pad],axis=1)
        else:
            # pad widths
            pad_width =pad_dim-w
            # pads
            pad =np.ones((h,pad_width,3))*pad_val
            # pad
            img =np.concatenate([img,pad],axis=1)
    else:
        # shape
        h,w,d=img.shape
        # pad heights
        if h>= pad_dim:
            return img 
        else:
            pad_height =pad_dim-h
            # pads
            pad =np.ones((pad_height,w,3))*pad_val
            # pad
            img =np.concatenate([img,pad],axis=0)
    return img.astype("uint8")    
#---------------------------------------------------------------
def correctPadding(img,dim,ptype="central",pvalue=255):
    '''
        corrects an image padding 
        args:
            img     :       numpy array of single channel image
            dim     :       tuple of desired img_height,img_width
            ptype   :       type of padding (central,left)
            pvalue  :       the value to pad
        returns:
            correctly padded image

    '''
    img_height,img_width=dim
    mask=0
    # check for pad
    h,w,d=img.shape
    
    w_new=int(img_height* w/h) 
    img=cv2.resize(img,(w_new,img_height),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
    h,w,d=img.shape
    if w > img_width:
        # for larger width
        h_new= int(img_width* h/w) 
        img=cv2.resize(img,(img_width,h_new),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
        # pad
        img=padWordImage(img,
                     pad_loc="tb",
                     pad_dim=img_height,
                     pad_type=ptype,
                     pad_val=pvalue)
        mask=img_width

    elif w < img_width:
        # pad
        img=padWordImage(img,
                    pad_loc="lr",
                    pad_dim=img_width,
                    pad_type=ptype,
                    pad_val=pvalue)
        mask=w
    
    # error avoid
    img=cv2.resize(img,(img_width,img_height),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
    return img,mask 
#---------------------------------------------------------------
def processImages(df,img_dim,ptype="left",factor=32):
    '''
        process a specific dataframe with filename,word,graphemes and mode
        args:
            df      :   the dataframe to process
            img_dim :   tuple of (img_height,img_width)  
            ptype   :   type of padding to use
    '''
    img_height,img_width=img_dim
    masks=[]
    for idx in tqdm(range(len(df))):
        try:
            img_path    =   df.iloc[idx,0]
            img=cv2.imread(img_path)
            # correct padding
            img,imask=correctPadding(img,img_dim,ptype=ptype)
            # mask
            imask=math.ceil((imask/img_width)*(img_width//factor))
            mask=np.zeros((img_height//factor,img_width//factor))
            mask[:,:imask]=1
            mask=mask.flatten().tolist()
            mask=[int(i) for i in mask]
            cv2.imwrite(img_path,img)
            masks.append(mask)
        except Exception as e:
            LOG_INFO(e)
    df["mask"]=masks    
    return df

#---------------------------------------------------------------
def processLabels(df,vocab,max_len):
    '''
        processLabels:
        * divides: word to - unicodes,components
        e-->encoded
        p-->paded
        u-->unicode
        g-->grapheme components
        r-->raw with out start end
    '''
    GP=GraphemeParser(language=None)
    # process text
    ## components
    df.word=df.word.progress_apply(lambda x:str(x))
    df["components"]=df.word.progress_apply(lambda x:GP.process(x))
    df.dropna(inplace=True)
    df["eg_label"]=df.components.progress_apply(lambda x:encode_label(x,vocab))
    ### grapheme
    start_end_value=len(vocab)+1
    pad_value      =len(vocab)+2
    df["label"]=df.eg_label.progress_apply(lambda x:pad_label(x,max_len,pad_value,start_end_value))
    return df 

#------------------------------------------------
def processData(csv,vocab,max_len,img_dim):
    '''
        processes the dataset
        args:
            csv         :   a csv file that contains filepath,word,source data
            vocab       :   language class
            max_len     :   model max_len
            img_dim     :   tuple of (img_height,img_width) 
            num_folds   :   creating folds of the data
    '''
    df=pd.read_csv(csv)
    # images
    df=processImages(df,img_dim)
    # labels
    df=processLabels(df,vocab,max_len)
    # save data
    cols=["filepath","mask","label"]
    df=df[cols]
    df.dropna(inplace=True)
    df.to_csv(csv,index=False)
    LOG_INFO(f"Not Found:{not_found}")
    return df