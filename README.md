# Project-idi-rnnchatbot

We want to improve the intelligence of a chatbot by using Grid-LSTM cells instead of ordinary LSTM cells. The paper by Kalchbrenner[1] describes how Grid-LSTM cells imporves the performance of a translation model. The translation model has an encoder-decoder architecture which we can use for a chatbot as well. 

Clone the repository and follow the instructions. 
Our folder structure is:
```sh
Parent folder
    |
    |-- project-idi-rnn (Our repository)
    |       |
    |       |-- Models
    |       |
    |       |-- Preprocessing
    |
    |-- ubuntu-ranking-dataset-creator (Where we get our data from)
```

### Download the Ubuntu Dialogue Corpus
If you want to train the model with the whole dataset, clone the repository you find [here](https://github.com/rkadlec/ubuntu-ranking-dataset-creator) from the Ubuntu Dialogue Corpus.
```sh
git clone https://github.com/rkadlec/ubuntu-ranking-dataset-creator.git
```


Navigate to the src folder, and run:
```sh
./generate.sh
```
NB! This is VERY time consuming. This command will start to download a tgz file, and further unpack the content into a dialogs folder. The dialogs requires ~8 GB of disk space.


### Create files
Make sure you the folder structure as described above.
In the Preprocessing folder, run:
```sh
python preprocess.py
```
This will maybe take around 20-30 minutes if you do not change the parameters, i.e. it will read all the files. 
OBS! You may get the error that the file is not found. You may have to turn of the file shield in you Anti Virus.

### Start training a model
In the Models folder, run:
```sh
python Ola_GridLSTM.py
```

### Chat with Ola
In the Models folder, run:
```sh
python Ola_GridLSTM.py --decode

```


### Results
##### Baseline
Our baseline is based on the encoder-decoder model described in Sutskever [3] and Vinyals [4]. 
TBA

##### GridLSTM
Our model is based on the paper by Kalckbrenner et al. [1] using Tensorflows Grid cells. 
TBA

### TODOs
 - If someone runs a model, and the files are not there, the model should preprocess the files. If the ubuntu raw files are not there, it should download the .tsv files.
 - Generate a small vocabulary and the necessary train, validate and test sets, with a size that github can handle
 - Compare Baseline and GridLSTM
 - Save the best checkpoints based on perplexity


### References
[1] [Kalchbrenner et al 2016](https://arxiv.org/pdf/1507.01526.pdf)
[2] [Ubuntu Dialogue Corpus](https://github.com/rkadlec/ubuntu-ranking-dataset-creator)
[3] [Sutskever](https://arxiv.org/abs/1409.3215)
[4] [Vinyals](http://arxiv.org/abs/1506.05869)