# Project-idi-rnnchatbot

We want to improve the intelligence of a chatbot by using Grid-LSTM cells instead of ordinary LSTM cells. The paper by Kalchbrenner[1] describes how Grid-LSTM cells imporves the performance of a translation model. The translation model has an encoder-decoder architecture which we can use for a chatbot as well. 

Clone the repository and follow the instructions. 

### Download the Ubuntu Dialogue Corpus
If you want to train the modesl with the whole dataset, download the Ubuntu Dialogue Corpus from [here](https://github.com/rkadlec/ubuntu-ranking-dataset-creator). 
NB! This is VERY time consuming.

Navigate to the src folder, and run:
```sh
./generate.sh
```
This command will start to download a tgz file, and further unpack the content into a dialogs folder. 
Our folder structure is:
```sh
Parent folder
    |
    |-- project-idi-rnn
    |       |
    |       |-- Models
    |       |
    |       |-- Preprocessing
    |
    |-- ubuntu-ranking-dataset-creator
```


### Create files
Make sure you the folder structure as described above. In the Preprocessing folder, run:
```sh
python preprocess.py
```

### Start training a model
In the Models folder, run:
```sh
python Ola_GridLSTM.py
```

### Chat with Ola:
```sh
python Ola_GridLSTM.py --decode

```


### Results
##### Baseline
TBA
##### GridLSTM
TBA

### TODOs
 - If someone runs a model, and the files are not there, the model should preprocess the files. If the ubuntu raw files are not there, it should download the .tsv files.
 - Generate a small vocabulary and the necessary train, validate and test sets, with a size that github can handle
 - Change folder structure as described above.. 


### References
[1] [Kalchbrenner et al 2016](https://arxiv.org/pdf/1507.01526.pdf)
[2] [Ubuntu Dialogue Corpus](https://github.com/rkadlec/ubuntu-ranking-dataset-creator)
