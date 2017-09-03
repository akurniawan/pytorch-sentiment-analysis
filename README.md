# pytorch-rnn-sentiment-analysis

## Description

Just assume this is my toy for learning pytorch for the first time (it's easy and definitely awesome!). In this repo you can found the implementation of char-rnn to do sentiment analysis based on twitter data. <br />

## Implementation Details

1. Implemented using LSTM cell for each charactesr
2. Each batches will be grouped in respect of their lengths

## Implementation Limitations

1. For current implementation, the dataset is set to tokenize the input based on characters. There is still no way to update the tokenization via config.
2. Still no way to update RNN cell from config.
3. Still no way to update optimizer from config.

## Dataset

You can download the data from [1]. It contains 1,578,627 classified tweets, each row is classified as 1 for positive sentiment and 0 for negative sentiment. Kudos to [2] for providing the link to the data!

## Reference
[1] http://thinknook.com/wp-content/uploads/2012/09/Sentiment-Analysis-Dataset.zip <br />
[2] http://thinknook.com/twitter-sentiment-analysis-training-corpus-dataset-2012-09-22/ <br />
[3] https://karpathy.github.io/2015/05/21/rnn-effectiveness/