# Project-idi-rnnchatbot

We want to improve the intelligence of a chatbot by using Grid-LSTM cells instead of ordinary LSTM cells. The paper by Kalchbrenner[1] describes how Grid-LSTM cells imporves the performance of a translation model. The translation model has an encoder-decoder architecture which we can use for a chatbot as well. 

Clone the repository and follow the instructions. 

### Download the Ubuntu Dialogue Corpus
Download the Ubuntu Dialogue Corpus from [here](https://github.com/rkadlec/ubuntu-ranking-dataset-creator).
The downloaded files should be in the same folder as the project folder, not in the project folder, unless you want to change the path to Ubuntu Dialogue Corpus in the read_every_data_file_and_create_initial_files method in preprocess.py.

### Create files
In the Preprocessing folder, run:
$ python preprocess.py

### Start training a model
In the Models folder, run:
```sh
$ python Ola_GridLSTM.py
```

### Chat with Ola:
```sh
$ python Ola_GridLSTM.py --decode

```


### Results
##### Baseline
TBA
##### GridLSTM
TBA

### TODOs
 - If someone runs a model, and the files are not there, the model should preprocess the files. If the ubuntu raw files are not there, it should download the .tsv files.
 - 


### References
[1] [Kalchbrenner et al 2016](https://arxiv.org/pdf/1507.01526.pdf)
[2] [Ubuntu Dialogue Corpus](https://github.com/rkadlec/ubuntu-ranking-dataset-creator)
