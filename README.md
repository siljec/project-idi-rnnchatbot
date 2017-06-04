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
    |-- opensubtitles-parser
```

### Download the Ubuntu Dialogue Corpus
If you want to train the model with the UDC corpus, clone the repository you find [here](https://github.com/rkadlec/ubuntu-ranking-dataset-creator).
```sh
git clone https://github.com/rkadlec/ubuntu-ranking-dataset-creator.git
```


Navigate to the src folder, and run:
```sh
./generate.sh
```
NB! This is VERY time consuming. This command will start to download a tgz file, and further unpack the content into a dialogs folder. The dialogs requires ~8 GB of disk space.


### Download the Open Subtitles dataset
If you want to train the model with the Open Subtitles corpus, clone the repository you find [here](https://github.com/inikdom/opensubtitles-parser).

This is a smaller dataset and hence, is less time consuming than the UDC

### Create files
Make sure you have the folder structure as described above.
In the Preprocessing folder, run:
```sh
python preprocess.py
```
This will at most take around 30 minutes if you do not change the parameters, i.e. it will read all the files. 
OBS! You may get the error that the file is not found. You may have to turn off the file shield in you Anti Virus.

Adding the open_subtitle flag, will preprocess the OS dataset:

```sh
python preprocess.py --open_subtitles
```

Other flags are available to change the behavior of the preprocessing script:
- --context_full_turns, this will add the entire last output in the front of the training input to include context to the training.

### Start training a model
In the Models folder, run the desired model:

Grid LSTM cells
```sh
python GridLSTM.py
```

LSTM cells
```sh
python LSTM.py
```

GRU cells
```sh
python LSTM.py --use_lstm=false
```

Stateful model:
```sh
python LSTM_stateful.py
```

### Chat with the trained models

It is easy to chat with the trained model, just add a decode flag. Example:
```sh
python GridLSTM.py --decode
```


### Results
##### Baseline
Our baseline is based on the encoder-decoder model described in Sutskever [3] and Vinyals [4]. 
TBA

##### GridLSTM
Our model is based on the paper by Kalckbrenner et al. [1] using Tensorflows Grid cells. 



### References
[1] [Kalchbrenner et al 2016](https://arxiv.org/pdf/1507.01526.pdf)
[2] [Ubuntu Dialogue Corpus](https://github.com/rkadlec/ubuntu-ranking-dataset-creator)
[3] [Sutskever](https://arxiv.org/abs/1409.3215)
[4] [Vinyals](http://arxiv.org/abs/1506.05869)