# pytorch-rnn-sentiment-analysis

## Description

Just assume this is my toy for learning pytorch for the first time (it's easy and definitely awesome!). In this repo you can find the implementation of both char-rnn and word-rnn to do sentiment analysis based on twitter data.

Not only sentiment analysis, you can also use this project as a sentence classification with multiple classes. Just put your class ids on the csv and you're good to go!

## Implementation Details

1. You can choose between LSTM and CNN-LSTM for the character decoder
2. Each batches will be grouped in respect of their lengths

## Implementation Limitations

1. For current implementation, the dataset is set to tokenize the input based on characters. There is still no way to update the tokenization via config.
2. Still no way to update RNN cell from config.
3. Still no way to update optimizer from config.

## How to run?
Run `pip install -r requirements.txt`

Run `python run.py` with the following options
```
optional arguments:
  -h, --help            show this help message and exit
  --epochs EPOCHS       Number of epochs
  --dataset DATASET     Path for your training, validation and test dataset.
                        As this package uses torch text to load the data,
                        please follow the format by providing the path and
                        filename without its extension
  --batch_size BATCH_SIZE
                        The number of batch size for every step
  --log_interval LOG_INTERVAL
  --save_interval SAVE_INTERVAL
  --validation_interval VALIDATION_INTERVAL
  --char_level CHAR_LEVEL
                        Whether to use the model with character level or word
                        level embedding. Specify the option if you want to use
                        character level embedding
  --model_config MODEL_CONFIG
                        Location of model config
  --model_dir MODEL_DIR
                        Location to save the model
```

This is the example of how you can run it
```
python run.py --model_config config/cnn_rnn.yml --epochs 50 --model_dir sentiment-analysis.pt --dataset data/sentiment
```

## Dataset

You can download the raw data from [1]. It contains 1,578,627 classified tweets, each row is classified as 1 for positive sentiment and 0 for negative sentiment. Kudos to [2] for providing the link to the data!. However, the data provided by [1] have 4 columns, while on this code we only need the text and the sentiment only, you can convert the data first by grabbing the first and the last columns before feeding into the algorithm.
For the alternative, you can also download the data from [4], this contains the same number of data as the original, but I have already cleaned it up a bit and you can run the code without any further modification.

Want to run with your own data? No problem, create csv files for training and testing with two columsn, the first one being the sentiment and the second being the text. Don't forget to use the same name for both files and differentiate it with suffix `.train` and `.test`.

## Reference
[1] http://thinknook.com/wp-content/uploads/2012/09/Sentiment-Analysis-Dataset.zip <br />
[2] http://thinknook.com/twitter-sentiment-analysis-training-corpus-dataset-2012-09-22/ <br />
[3] https://karpathy.github.io/2015/05/21/rnn-effectiveness/ <br />
[4] https://drive.google.com/file/d/1Go7FXn4mpIgle1X2mO1xYPDO0ZgWuPI6/view
