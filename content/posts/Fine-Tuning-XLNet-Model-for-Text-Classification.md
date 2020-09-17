---
title: "Fine Tuning XLNet Model for Text Classification"
date: 2020-09-09T00:40:27+05:30
draft: false
#featuredImage: "huggingface.png"
featuredImagePreview: "/images/FineTuningXLnet.jpg"
#coverImage: "huggingface.png"
images: ["/images/FineTuningXLnet.jpg"]
tags: ["Deep Learning", "Transformers", "XLNet", "Text Classification"]
categories: ["Natural Language Understanding"]
description: "In this article, we will see how to fine tune a XLNet model on custom data, for text classification using Transformers🤗. XLNet is powerful! It beats BERT and its other variants in 20 different tasks. In simple words - XLNet is a generalized autoregressive model.

An Autoregressive model is a model which uses the context word to predict the next word. So, the next token is dependent on all previous tokens.

XLNET is generalized because it captures bi-directional context by means of a mechanism called permutation language modeling. 

It integrates the idea of auto-regressive models and bi-directional context modeling, yet overcoming the disadvantages of BERT and thus outperforming BERT on 20 tasks, often by a large margin in tasks such as question answering, natural language inference, sentiment analysis, and document ranking.


In this article, we will take a pretrained `XLNet` model and fine tune it on our dataset."
---
<!--more-->

{{< admonition type=abstract title="Abstract" open=True >}}
In this article, we will see how to fine tune a XLNet model on custom data, for text classification using **Transformers🤗**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KUJoHQU_19iav6Lxu0_Ay2qWUrOLiltN?usp=sharing)

{{< /admonition >}}

{{< figure src="/images/FineTuningXLnet.jpg" >}}

## Introduction
**XLNet** is powerful! It beats BERT and its other variants in 20 different tasks.

>The XLNet model was proposed in XLNet: [Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/abs/1906.08237) by Zhilin Yang*, Zihang Dai*, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov, Quoc V. Le. XLnet is an extension of the Transformer-XL model pre-trained using an autoregressive method to learn bidirectional contexts by maximizing the expected likelihood over all permutations of the input sequence factorization order.

In simple words - XLNet is a generalized autoregressive model.

An **Autoregressive model** is a model which uses the context word to predict the next word. So, the next token is dependent on all previous tokens.

XLNET is **generalized** because it captures bi-directional context by means of a mechanism called “permutation language modeling”. 

It integrates the idea of auto-regressive models and bi-directional context modeling, yet overcoming the disadvantages of BERT and thus outperforming BERT on 20 tasks, often by a large margin in tasks such as question answering, natural language inference, sentiment analysis, and document ranking.

**In this article, we will take a pretrained `XLNet` model and fine tune it on our dataset.**

So, let's talk about the dataset.

## Data
We will take a dataset from Kaggle's text classification challenge (Ongoing as of now) -  [Real or Not? NLP with Disaster Tweets](https://www.kaggle.com/c/nlp-getting-started/overview). 

In this competition, we have to build a machine learning model that predicts which Tweets are about real disasters and which one’s aren’t. It's a small dataset of 10,000 tweets that were hand classified.

We will use this data to fine tune a pretrained XLNet model.

## Let's Code
{{< admonition type=note title="Note" open=True >}}
We will use Colab notebook to write our code so that we can leverage GPU enabled environment.
{{< /admonition >}}

### Installing Dependencies
 - First, lets spin up a [Colab notebook](https://colab.research.google.com/). 
 - Download the data from [Real or Not? NLP with Disaster Tweets](https://www.kaggle.com/c/nlp-getting-started/data). You will have 3 files, `train.csv`, `test.csv` and `sample_submission.csv`
 - Upload it to your Colab Notebook session.
 - Install the latest stable `pytorch 1.6`, `transformers` and `simpletransformers`.


```python
! pip uninstall torch torchvision -y
! pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
!pip install -U transformers
!pip install -U simpletransformers  
```
 Now we're good to go.

### Preprocessing

```python

import pandas as pd
import numpy as np

df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

df_train.head()

```
{{< figure src="/images/XLNet-dataframe.png" >}}

***
We have 5 columns in our data:
- `id`: it is a unique identifier of tweets.
- `keyword`: It contains the keywords made on the tweets.
- `location`: The location the tweet was sent from.
- `text`: it is actual tweet made by the users
- `target`: Whether a given tweet is about a real disaster or not. If so, predict a 1. If not, predict a 0.
***

Let's look at the distribution of target class
```python
df_train.target.value_counts()
```
`0    4342`

`1    3271`

`Name: target, dtype: int64`

The dataset is pretty much balanced. We have 3271 tweets about disasters while 4342 tweets otherwise.
***
Let's have a look at the `keyword` and `location` columns
```python
print(f"Keyword column has {df_train.keyword.isnull().sum()/df_train.shape[0]*100}% null values")
print(f"Location column has {df_train.location.isnull().sum()/df_train.shape[0]*100}% null values)
```
`Keyword column has 0.80% null values`

`Location column has 33.27% null values`

`location` has 33% missing values while `keyword` has 0.8% null values. We will not delve into filling up missing values and will leave these columns as it is.

The `text` and `target` columns is of our interest.

***
Let's have a look at the `text` column

```python
df_train.sample(10)['text'].tolist()
```
`['Two giant cranes holding a bridge collapse into nearby homes http://t.co/jBJRg3eP1Q',`

 `"Apollo Brown - 'Detonate' f. M.O.P. | http://t.co/H1xiGcEn7F",`

 `'Listening to Blowers and Tuffers on the Aussie batting collapse at Trent Bridge reminds me why I love @bbctms! Wonderful stuff! #ENGvAUS',`

 `'Downtown Emergency Service Center is hiring! #Chemical #Dependency Counselor or Intern in #Seattle apply now! #jobs http://t.co/HhTwAyT4yo',`

 `'Car engulfed in flames backs up traffic at Parley\x89Ûªs Summit http://t.co/RmucfjCaZr',`

 `'After death of Palestinian toddler in arson\nattack Israel cracks down on Jewish',`

 `'Students at Sutherland remember Australian casualties at Lone Pine Gallipoli\n http://t.co/d50oRfXoFB via @theleadernews',`
 `'FedEx no longer to transport bioterror germs in wake of anthrax lab mishaps http://t.co/hrqCJdovJZ',`

 `'@newyorkcity for the #international emergency medicine conference w/ Lennox Hill hospital and #drjustinmazur',`

 `'My back is so sunburned :(']`

 We see that the text columns contains `#`, `@`, and `links` which needs to be cleaned.
 ***
 Let's write a simple function to clean up:
  - `#`
  - username starting with `@`
  - `links`

We will use `tweet-preprocessor` to do this. 

`tweet-preprocessor.clean()` function can help us get rid of irrelevant tokens such as any hashtags, @username or links from the tweet and make it super clean to feed into `XLNet` model.
 ```python
! pip install tweet-preprocessor
import preprocessor as p
from tqdm.notebook import tqdm
tqdm.pandas()

# function to clean @, #, and links from tweets
def clean_text(text):
  text = text.replace("#","")
  return p.clean(text)

# Appling function to train and test data
from tqdm.notebook import tqdm
tqdm.pandas()

df_train['clean_text'] = df_train['text'].astype(str).progress_map(clean_text)
df_test['clean_text'] = df_test['text'].astype(str).progress_map(clean_text)

```
`100%`
`7613/7613 [00:49<00:00, 154.19it/s]`

`100%`
`3263/3263 [00:48<00:00, 67.34it/s]`

Now, we have clean text in `clean_text` column.
***
Now, let's split our data into `train` and `eval` set

```python
# splitting the data into training and eval dataset
X = df_train['clean_text']
y = df_train['target']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

train_df = pd.DataFrame(X_train)
train_df['target'] = y_train

eval_df = pd.DataFrame(X_test)
eval_df['target'] = y_test

train_df.shape, eval_df.shape

```

`((6090, 2), (1523, 2))`

We divided our data into `train_df` and `eval_df` in 80:20 startified split.
We have 6090 tweets for training and 1523 tweets for evaluation.

Now, we are all set for training `XLNet`.

### XLNet Training

For training `XLNet`, we will use `simpletransformers` which is super easy to use library built on top of our beloved `transformers`.

`simpletransformers` has a unified functions to train any SOTA pretrained NLP model available in `transformers`.
So you get the power of SOTA pretrained language models like `BERT` and its variants, `XLNet`, `ELECTRA`, `T5` etc. wrapped in easy to use functions.

As you see below, it just takes 3 lines of code to train a `XLNet` model. And the same holds true for training it from scratch or just fine tuning the model on custom dataset.

I have kept `num_train_epochs: 4`, `train_batch_size: 32` and `max_seq_length: 128` - so that it fits into Colab compute limits. Feel free to play with a lot of parameters mentioned in `args` in the code below.
```python
# We will import ClassificationModel - as we need to solve binary text classification
from simpletransformers.classification import ClassificationModel
import pandas as pd
import logging
import sklearn


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# They are lot of arguments to play with
'''
args = {
   'output_dir': 'outputs/',
   'cache_dir': 'cache/',
   'fp16': True,
   'fp16_opt_level': 'O1',
   'max_seq_length': 256,
   'train_batch_size': 8,
   'eval_batch_size': 8,
   'gradient_accumulation_steps': 1,
   'num_train_epochs': 3,
   'weight_decay': 0,
   'learning_rate': 4e-5,
   'adam_epsilon': 1e-8,
   'warmup_ratio': 0.06,
   'warmup_steps': 0,
   'max_grad_norm': 1.0,
   'logging_steps': 50,
   'evaluate_during_training': False,
   'save_steps': 2000,
   'eval_all_checkpoints': True,
   'use_tensorboard': True,
   'overwrite_output_dir': True,
   'reprocess_input_data': False,
}

'''

# Create a ClassificationModel
model = ClassificationModel('xlnet', 'xlnet-base-cased', args={'num_train_epochs':4, 'train_batch_size':32, 'max_seq_length':128}) # You can set class weights by using the optional weight argument

# Train the model
model.train_model(train_df)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(eval_df, acc=sklearn.metrics.accuracy_score)

```
`Downloading: 100%
760/760 [00:10<00:00, 71.0B/s]`

`Downloading: 100%
467M/467M [00:10<00:00, 45.2MB/s]`

`Downloading: 100%
798k/798k [00:14<00:00, 56.1kB/s]`

`100%
6090/6090 [08:15<00:00, 12.29it/s]`

`Epoch 4 of 4: 100%
4/4 [08:12<00:00, 123.24s/it]`

`Epochs 0/4. Running Loss: 0.4059: 100%
191/191 [08:12<00:00, 2.58s/it]`

`Epochs 1/4. Running Loss: 0.2305: 100%
191/191 [02:01<00:00, 1.57it/s]`

`Epochs 2/4. Running Loss: 0.4360: 100%
191/191 [04:24<00:00, 1.38s/it]`

`Epochs 3/4. Running Loss: 0.0260: 100%
191/191 [02:28<00:00, 1.28it/s]`

`100%
1523/1523 [00:23<00:00, 65.14it/s]`

`Running Evaluation: 100%
191/191 [00:20<00:00, 9.17it/s]`

`INFO:simpletransformers.classification.classification_model:{'mcc': 0.6457675302369492, 'tp': 518, 'tn': 741, 'fp': 128, 'fn': 136, 'acc': 0.8266579120157583, 'eval_loss': 0.5341164009543184}`
***

We have achieved a decent accuracy of 82.6% on our eval set. This accracy is just out of the box - means with **no feature engineering**, with **no hyparameter-tuning**. **Just out of the box!**  

## 🥳 We're Done!
Let's submit the predictions to Kaggle and see where we stand.
```python
predictions, raw_outputs = model.predict(df_test.clean_text.tolist())
sample_sub=pd.read_csv("sample_submission.csv")
sample_sub['target'] = predictions

sample_sub.to_csv("submission_09092020_xlnet_base.csv", index=False)
```
`INFO:simpletransformers.classification.classification_model: Converting to features started. Cache is not used.`

`100%
3263/3263 [00:01<00:00, 3216.81it/s]`

`100%`
`408/408 [00:38<00:00, 10.68it/s]`
***



{{< figure src="/images/xlnet-kaggle.png" >}}
We're in top 18%. It's a good start considering `XLNet` out of the box performance - with no feature engineering at all.

Now, we have a decent baseline to improve our model upon.


## Notebooks
{{< admonition type=success title="Attachments" open=True >}}
- [Go to Dataset](https://www.kaggle.com/c/nlp-getting-started/overview). 
- [Go to Google Colab Notebook](https://colab.research.google.com/drive/1KUJoHQU_19iav6Lxu0_Ay2qWUrOLiltN?usp=sharing)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KUJoHQU_19iav6Lxu0_Ay2qWUrOLiltN?usp=sharing)
{{< /admonition >}}

***

## Subscribe
Thank you for reading my blog! 🤗

If you like what you read, 🚀[Subscribe to get notified of new blog posts.](https://tinyletter.com/shivanandroy) 