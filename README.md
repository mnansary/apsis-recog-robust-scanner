
# synthetic words

```python
Version: 0.0.1     
```
### **Related resources**:


**LOCAL ENVIRONMENT**  
```python
OS          : Ubuntu 18.04.3 LTS (64-bit) Bionic Beaver        
Memory      : 7.7 GiB  
Processor   : Intel® Core™ i5-8250U CPU @ 1.60GHz × 8    
Graphics    : Intel® UHD Graphics 620 (Kabylake GT2)  
Gnome       : 3.28.2  
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


# TODO
- ```conda activate your_env```
- ```cd scripts```
- ```python datagen.py data_dir identifier```

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
                        the vocabulary to use. available: english_numbers,bangla_numbers,english_all,bangla_all
  --tf_size TF_SIZE     the size of data to store in 1 tfrecord:default=128
  --img_height IMG_HEIGHT
                        height for each grapheme: default=64
  --img_width IMG_WIDTH
                        width for each grapheme: default=512

```

