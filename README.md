
# Robust Scanner 

**Tensorflow-2 implementation with TPU support**

```python
Version: 0.0.2  
Author : MD.Nazmuddoha Ansary
```
### **Related resources**:


**LOCAL ENVIRONMENT**  
```python
OS          : Ubuntu 20.04.3 LTS       
Memory      : 23.4 GiB 
Processor   : Intel® Core™ i5-8250U CPU @ 1.60GHz × 8    
Graphics    : Intel® UHD Graphics 620 (Kabylake GT2)  
Gnome       : 3.36.8 
```

**python requirements**
* **pip requirements**: ```pip install -r requirements.txt``` 
> Its better to use a virtual environment 
OR use conda-
* **conda**: use environment.yml: ```conda env create -f environment.yml```


# dataset requirements

create a dataset as follows:
 
```
dataset_folder
    |-images
        |-image_n.png
        |-xyz_n.png
        |-...........
        |-...........

    |-data.csv

```
* data.csv colums:
    * filepath: /home/something/something/dataset_folder/images/imgxyz.png
    * word    : text (ground_truth)


# Execution
- ```conda activate your_env```
- ```cd scripts```
- ```python datagen.py data_dir identifier --seq_max_len 40 --vocab_iden all --tf_size 10240```

```python

usage: Recognizer Dataset Creating Script [-h] [--seq_max_len SEQ_MAX_LEN] [--vocab_iden VOCAB_IDEN] [--tf_size TF_SIZE] [--img_height IMG_HEIGHT] [--img_width IMG_WIDTH] data_dir iden

positional arguments:
  data_dir              Path of the source data folder that contains langauge datasets
  iden                  identifier to identify the dataset

optional arguments:
  -h, --help            show this help message and exit
  --seq_max_len SEQ_MAX_LEN
                        the maximum length of data for modeling
  --vocab_iden VOCAB_IDEN
                        the vocabulary to use. available: english_numbers,bangla_numbers,english_all,bangla_all,all
  --tf_size TF_SIZE     the size of data to store in 1 tfrecord:default=128
  --img_height IMG_HEIGHT
                        height for each grapheme: default=64
  --img_width IMG_WIDTH
                        width for each grapheme: default=512

```

- upon successful execution a config.json file will be created in the repo directory. 
- use this config.json file while training.
- zip the tfrecords folder and upload it to a kaggle-dataset (public)
- the kaggle dataset should have the following structre:

```
* dataset ("/input/dataset/)
|-dataset_iden
 |-dataset_iden
  |-x.tfrecord
  |-x.tfrecord
  |-x.tfrecord
  .................
|-config.json
|-enc.h5
|-seq.h5
|-pos.h5
|-fuse.h5
```

- if no pretrained models are provided: use_pretrained=False while training

- use **notebooks/train.ipynb** for training with TPU