# Project-idi-rnnchatbot

This repository consists of six different RNN architectures to create chatbots. All of the models are build upon the sequence-to-sequence model from Tensorflow [1]. LSTM, GRU and Grid LSTM [2] is the explored cells. For conversation context purposes, we developed the Stateful and Context-Prepro models. Last, we also created chatbots without the use of Tensorflows efficiency buckets.

To use this project, clone the repository and follow the instructions. 
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
If you want to train the model with the UDC corpus [3], clone the repository you find [here](https://github.com/rkadlec/ubuntu-ranking-dataset-creator).
```sh
git clone https://github.com/rkadlec/ubuntu-ranking-dataset-creator.git
```


Navigate to the src folder, and run:
```sh
./generate.sh
```
NB! This is VERY time-consuming. This command will start to download a tgz file, and further, unpack the content into a dialogs folder. The dialogs require ~8 GB of disk space.


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
OBS! You may get the error that the file is not found. You may have to turn off the file shield in you Anti-Virus.

Adding the open_subtitle flag will preprocess the OS dataset:

```sh
python preprocess.py --open_subtitles
```

Other flags are available to change the behavior of the preprocessing script:
- --context_full_turns, this will add the entire last output in the front of the training input to include context to the training.

### Start training a model
By default, the models will train on the UDC. To train on the OS dataset, simply add the flag (--opensubtitles) to the command. In the Models folder, run the desired model:

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

One-Bucket model:
```sh
python LSTM.py --one_bucket
```

Stateful model:
```sh
python LSTM_stateful.py
```

Context-Prepro model:
```sh
python preprocessing --contextFullTurns
python python LSTM.py --contextFullTurns
```



### Chat with the trained models

It is easy to chat with the trained model, just add a decode flag. Example:
```sh
python GridLSTM.py --decode
```


### Results

Our results are based on human evaluations. The UCD results are based on 30 participants responses, while the OS results are gathered from 50 evaluators opinions.
 
#### UDC results
| Model          | Grammar | Content | Total score |
|----------------|---------|---------|-------------|
|Dataset         |   3.86  |   3.69  |    3.77     |
|Grid LSTM       |   3.59  |   3.00  |    3.29     |
|LSTM            |   3.61  |   3.01  |    3.31     |
|GRU             |   3.45  |   2.91  |    3.18     | 


| Model          | Grammar | Content | Total score |
|----------------|---------|---------|-------------|
|Dataset         |   4.23  |   3.98  |    4.1      |
|Stateful        |   3.80  |   2.71  |    3.25     |
|LSTM            |   3.78  |   2.38  |    3.08     |
|Context-Prepro  |   3.75  |   2.08  |    2.92     | 


#### OS results

| Model          | Grammar | Content | Total score |
|----------------|---------|---------|-------------|
|LSTM            |   3.91  |   2.67  |    3.29     |
|Grid LSTM       |   4.14  |   3.26  |    3.70     |
|Stateful-Decoder|   3.97  |   2.67  |    3.32     |
|One-Bucket      |   3.80  |   2.78  |    3.29     | 

### Conclusion

The Grid LSTM slightly outperforms the LSTM model, while the Stateful model shows that it can handle the content of a conversation better than the other models.


### References
[1] [Tensorflow Sequence-to-sequence](https://www.tensorflow.org/tutorials/seq2seq)
[2] [Kalchbrenner et al 2016](https://arxiv.org/pdf/1507.01526.pdf)
[3] [Ubuntu Dialogue Corpus](https://github.com/rkadlec/ubuntu-ranking-dataset-creator)
[4] [Sutskever](https://arxiv.org/abs/1409.3215)
[5] [Vinyals](http://arxiv.org/abs/1506.05869)