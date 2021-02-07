---
title: "Top 10 Most Useful But Underrated Python Libraries for Data Science"
date: 2020-10-18T00:42:27+05:30
draft: false
#featuredImage: "/posts/ml/images/resumeanalyzer.png"
featuredImagePreview: "/posts/ml/images/underratedpythonlibrary.jpg"
coverImage: "/posts/ml/images/underratedpythonlibrary.jpg"
images: ["/posts/ml/images/underratedpythonlibrary.jpg"]
tags: ["Data Science", "Machine Learning", "Python Packages"]
categories: ["Python Packages"]
description: "Top 10 most useful but underrated python libraries for data science and machine learning"
---
<!--more-->


{{< figure src="/posts/ml/images/underratedpythonlibrary.jpg" >}}






## Introduction
Python community offers jillion of python packages for data science pipeline starting from data cleaning to building deep learning models to deployment. Most appreciated and commonly used packages are - 
- `Pandas`  : Data manipulation and analysis
- `Matplotlib/Seaborn/Plotly ` :  Data visualization
- `Scikit-learn`  :  Building Machine Learning models
- `Keras/Tensorflow/Pytorch`  :  building deep learning models
- `Flask ` :  Web app development/ML Applications


These packages have received their appreciation and love from the data science community.

But there are some python libraries in data science that are useful but underrated at the same time. 
These packages can save you from writing a lot of code. 

They give you the ease of using state of the art models in one just single line of code.

Let's dive in.









### 1. `MissingNo`

[`MissingNo`](https://github.com/ResidentMario/missingno) is a python library for null value or missing values analysis with impressive visualization like data display, bar charts, heatmaps and dendograms.
 - **Installation**: `pip install missingno`
 - **Github**: [`MissingNo`](https://github.com/ResidentMario/missingno)

```python
# pip install missingno
import missingno as msno

# missing value visualization: dense data display
msno.matrix(dataframe)

# missing value visualization: bar charts
msno.bar(dataframe)

# missing value visualization: heatmaps
msno.heatmap(dataframe)

# missing value visualization: Dendogram
msno.dendrogram(dataframe)
```




### 2. `MLExtend`
[`MLExtend`](http://rasbt.github.io/mlxtend/) stands for Machine Learning Extensions and is created by [Sebastian Raschka](http://sebastianraschka.com/). 

As the name suggests, it extends the current implementation of many machine learning algorithms, makes it more useful to use and definitely saves a lot of time. 

For e.g. Association Rule Mining (with Apriori, Fpgrowth & Fpmax support), EnsembleVoteClassifier, StackingCVClassifier


- **Installation**: `pip install mlxtend`
- **Github**: [`MlExtend`](https://github.com/rasbt/mlxtend)

For e.g. `StackingCVClassifier` can be implemented as
 ```python
 from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import StackingCVClassifier
import numpy as np
import warnings

warnings.simplefilter('ignore')

RANDOM_SEED = 42

clf1 = KNeighborsClassifier(n_neighbors=1)
clf2 = RandomForestClassifier(random_state=RANDOM_SEED)
clf3 = GaussianNB()
lr = LogisticRegression()

# Starting from v0.16.0, StackingCVRegressor supports
# `random_state` to get deterministic result.
sclf = StackingCVClassifier(classifiers=[clf1, clf2, clf3],
                            meta_classifier=lr,
                            random_state=RANDOM_SEED)

print('3-fold cross validation:\n')

for clf, label in zip([clf1, clf2, clf3, sclf], 
                      ['KNN', 
                       'Random Forest', 
                       'Naive Bayes',
                       'StackingClassifier']):

    scores = model_selection.cross_val_score(clf, X, y, 
                                              cv=3, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" 
          % (scores.mean(), scores.std(), label))
```







### 3.  `Flair`

[`Flair`](https://github.com/flairNLP/flair) is a powerful NLP library  which allows you to apply our state-of-the-art natural language processing (NLP) models to your text, such as named entity recognition (NER), part-of-speech tagging (PoS), sense disambiguation and classification.

- **Installation**: `pip install flair`
- **Github**: [`Flair`](https://github.com/flairNLP/flair)

Yes  -  You have many libraries which promises that  - What sets Flair apart?

It's Stacked embeddings!

Stacked embeddings is one of the most interesting features of Flair which will make you use this library even more.

They provide means to combine different embeddings together. You can use both traditional word embeddings (like GloVe, word2vec, ELMo) together with Flair contextual string embeddings or BERT.

You can very easily mix and match Flair, ELMo, BERT and classic word embeddings. All you need to do is instantiate each embedding you wish to combine and use them in a `StackedEmbedding`. 

For instance, let's say we want to combine the multilingual Flair and BERT embeddings to train a hyper-powerful multilingual downstream task model. First, instantiate the embeddings you wish to combine:

```python
from flair.embeddings import FlairEmbeddings, BertEmbeddings

# init Flair embeddings
flair_forward_embedding = FlairEmbeddings('multi-forward')
flair_backward_embedding = FlairEmbeddings('multi-backward')

# init multilingual BERT
bert_embedding = BertEmbeddings('bert-base-multilingual-cased')
from flair.embeddings import StackedEmbeddings

# now create the StackedEmbedding object that combines all embeddings
stacked_embeddings = StackedEmbeddings(
    embeddings=[flair_forward_embedding, flair_backward_embedding, bert_embedding])

```



Now just use this embedding like all the other embeddings, i.e. call the `embed()` method over your sentences.

```python
sentence = Sentence('The grass is green .')

# just embed a sentence using the StackedEmbedding 
# as you would with any single embedding.
stacked_embeddings.embed(sentence)

# now check out the embedded tokens.
for token in sentence:
    print(token)
    print(token.embedding)

```




### 4. `AdaptNLP`
[`AdaptNLP`](https://github.com/Novetta/adaptnlp) is another easy to use but powerful NLP toolkit built on top of `Flair` and `Transformers` for running, training and deploying state of the art deep learning models. 

It has a unified API for end to end NLP tasks: Token tagging, Text Classification, Question Anaswering, Embeddings, Translation, Text Generation etc.

- **Installation**: `pip install adaptnlp`
- **Github**: [`AdaptNLP`](https://github.com/Novetta/adaptnlp)

Sample code for **Question Answering**
```python
from adaptnlp import EasyQuestionAnswering 
from pprint import pprint

## Example Query and Context 
query = "What is the meaning of life?"
context = "Machine Learning is the meaning of life."
top_n = 5

## Load the QA module and run inference on results 
qa = EasyQuestionAnswering()
best_answer, best_n_answers = qa.predict_qa(query=query, context=context, n_best_size=top_n, mini_batch_size=1, model_name_or_path="distilbert-base-uncased-distilled-squad")

## Output top answer as well as top 5 answers
print(best_answer)
pprint(best_n_answers)

```

Sample code for **Summarization**
```python
from adaptnlp import EasySummarizer

# Text from encyclopedia Britannica on Einstein
text = """Einstein would write that two “wonders” deeply affected his early years. The first was his encounter with a compass at age five. 
          He was mystified that invisible forces could deflect the needle. This would lead to a lifelong fascination with invisible forces. 
          The second wonder came at age 12 when he discovered a book of geometry, which he devoured, calling it his 'sacred little geometry 
          book'. Einstein became deeply religious at age 12, even composing several songs in praise of God and chanting religious songs on 
          the way to school. This began to change, however, after he read science books that contradicted his religious beliefs. This challenge 
          to established authority left a deep and lasting impression. At the Luitpold Gymnasium, Einstein often felt out of place and victimized 
          by a Prussian-style educational system that seemed to stifle originality and creativity. One teacher even told him that he would 
          never amount to anything."""

summarizer = EasySummarizer()


# Summarize
summaries = summarizer.summarize(text = text, model_name_or_path="t5-small", mini_batch_size=1, num_beams = 4, min_length=0, max_length=100, early_stopping=True)

print("Summaries:\n")
for s in summaries:
    print(s, "\n")
```



### 5. `SimpleTransformers`
[`SimpleTransformers`](https://github.com/ThilinaRajapakse/simpletransformers) is awesome and my go to library for any NLP deep learning models. It packs all the powerful features of Huggingface's `transformers` in just 3 lines of code for end to end NLP tasks.

- **Installation**: `pip install simpletransformers`
- **Github**: [`SimpleTransformers`](https://github.com/ThilinaRajapakse/simpletransformers)



Sample code for **Text Classification**

```python
from simpletransformers.classification import ClassificationModel
import pandas as pd
import logging


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# Train and Evaluation data needs to be in a Pandas Dataframe of two columns. The first column is the text with type str, and the second column is the label with type int.
train_data = [['Example sentence belonging to class 1', 1], ['Example sentence belonging to class 0', 0]]
train_df = pd.DataFrame(train_data)

eval_data = [['Example eval sentence belonging to class 1', 1], ['Example eval sentence belonging to class 0', 0]]
eval_df = pd.DataFrame(eval_data)

# Create a ClassificationModel
model = ClassificationModel('roberta', 'roberta-base') # You can set class weights by using the optional weight argument

# Train the model
model.train_model(train_df)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(eval_df)
```


Sample code for **Language Model Training**
```python
from simpletransformers.language_modeling import LanguageModelingModel
import logging


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

train_args = {
    "reprocess_input_data": True,
    "overwrite_output_dir": True,
}

model = LanguageModelingModel('bert', 'bert-base-cased', args=train_args)

model.train_model("wikitext-2/wiki.train.tokens", eval_file="wikitext-2/wiki.test.tokens")

model.eval_model("wikitext-2/wiki.test.tokens")
```



### 6. `Sentence-Transformers`
[`Sentence-Transformers`](https://github.com/UKPLab/sentence-transformers) is a python package to compute the dense vector representations of sentences or paragraphs. 

- **Installation**: `pip install -U sentence-transformers`
- **Github**: [`Sentence-Transformers`](https://github.com/UKPLab/sentence-transformers)

This library not only allows to generate embeddings from [state of the art pretrained transformer models](https://www.sbert.net/docs/pretrained_models.html) but also allows embeddings after fine-tuning on your custom dataset.

These embeddings are useful for various downstream tasks like semantic search or clustering



Sample code for computing **Embeddings**

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('distilbert-base-nli-mean-tokens')

sentences = ['This framework generates embeddings for each input sentence',
    'Sentences are passed as a list of string.', 
    'The quick brown fox jumps over the lazy dog.']
sentence_embeddings = model.encode(sentences)

for sentence, embedding in zip(sentences, sentence_embeddings):
    print("Sentence:", sentence)
    print("Embedding:", embedding)
    print("")

```


Sample code for **Semantic Search**
```python
from sentence_transformers import SentenceTransformer, util
import torch

embedder = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

# Corpus with example sentences
corpus = ['A man is eating food.',
          'A man is eating a piece of bread.',
          'The girl is carrying a baby.',
          'A man is riding a horse.',
          'A woman is playing violin.',
          'Two men pushed carts through the woods.',
          'A man is riding a white horse on an enclosed ground.',
          'A monkey is playing drums.',
          'A cheetah is running behind its prey.'
          ]
corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)

# Query sentences:
queries = ['A man is eating pasta.', 'Someone in a gorilla costume is playing a set of drums.', 'A cheetah chases prey on across a field.']


# Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
top_k = 5
for query in queries:
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
    cos_scores = cos_scores.cpu()

    #We use torch.topk to find the highest 5 scores
    top_results = torch.topk(cos_scores, k=top_k)

    print("\n\n======================\n\n")
    print("Query:", query)
    print("\nTop 5 most similar sentences in corpus:")

    for score, idx in zip(top_results[0], top_results[1]):
        print(corpus[idx], "(Score: %.4f)" % (score))
```








### 7. `Tweet-Preprocessor`
Preprocessing social data can be a bit frustrating at times bacause of irrelevant elements within the text like links, emojis, hashtags, usernames, mentions etc..

But not any more!

`Tweet-Preprocessor` is for you. This library cleans your text/tweets in a single line of code.

- **Installation**: `pip install tweet-preprocessor`
- **Github**: [`Tweet-Preprocessor`](https://github.com/s/preprocessor)



```python
>>> import preprocessor as p
>>> p.clean('Preprocessor is #awesome 👍 https://github.com/s/preprocessor')

# output: 'Preprocessor is'
```



Currently supports cleaning, tokenizing and parsing URLs, Hashtags,  Mentions, Reserved words (RT, FAV), Emojis, SmileysNumbers and you have full control over what you want to clean from the text.




### 8. `Gradio`
[`Gradio`](https://github.com/gradio-app/gradio) is another super cool library to quickly create customizable UI components to demo your ML/DL models within your jupyter notebook or in the browser.

- **Installation**: `pip install gradio`
- **Github**: [`Gradio`](https://github.com/gradio-app/gradio)


{{< figure src="/posts/ml/images/gradio.gif" title="Source: Gradio">}}

### 9. `PPScore`

`PPScore` is Predictive Power Score which is a better alternative than `df.corr()` to find the correlation or relationship between 2 variables.

- **Installation**: `pip install -U ppscore`
- **Github**: [`PPScore`](https://github.com/8080labs/ppscore)

Why is it better than correlation?

- It detects both linear and non-linear relationship between 2 columns
- It gives a normalized score ranging from 0 (no predictive power) to 1 (perfect predictive power)
- It takes both numeric and categorical variables as input, so no need to convert your categorical variables into dummy variables before feeding it to PPScore.

```python
import ppscore as pps

# Based on the dataframe we can calculate the PPS of x predicting y
pps.score(df, "x", "y")

# We can calculate the PPS of all the predictors in the dataframe against a target y
pps.predictors(df, "y")
```



### 10. `Pytorch-Forecasting`
[`Pytorch-Forecasting`](https://github.com/jdb78/pytorch-forecasting) is python toolkit built on top of `pytorch-lightening` which aims to solve time series forecasting with neural networks with ease.

- **Installation**: `pip install pytorch-forecasting`
- **Github**: [`Pytorch-Forecasting`](https://github.com/jdb78/pytorch-forecasting)

This library provides abstraction over handling missing values, variable transformation, Tensorboard support, prediction & dependency plots, Range optimizer for faster training and Optuna for hyperparamter tuning.



Usage script: [Pytorch-Forecasting](https://github.com/jdb78/pytorch-forecasting)

```python
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer

# load data
data = ...

# define dataset
max_encode_length = 36
max_prediction_length = 6
training_cutoff = "YYYY-MM-DD"  # day for cutoff

training = TimeSeriesDataSet(
    data[lambda x: x.date <= training_cutoff],
    time_idx= ...,
    target= ...,
    group_ids=[ ... ],
    max_encode_length=max_encode_length,
    max_prediction_length=max_prediction_length,
    static_categoricals=[ ... ],
    static_reals=[ ... ],
    time_varying_known_categoricals=[ ... ],
    time_varying_known_reals=[ ... ],
    time_varying_unknown_categoricals=[ ... ],
    time_varying_unknown_reals=[ ... ],
)


validation = TimeSeriesDataSet.from_dataset(training, data, min_prediction_idx=training.index.time.max() + 1, stop_randomization=True)
batch_size = 128
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=2)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=2)


early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=1, verbose=False, mode="min")
lr_logger = LearningRateMonitor()
trainer = pl.Trainer(
    max_epochs=100,
    gpus=0,
    gradient_clip_val=0.1,
    limit_train_batches=30,
    callbacks=[lr_logger, early_stop_callback],
)


tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,
    hidden_size=32,
    attention_head_size=1,
    dropout=0.1,
    hidden_continuous_size=16,
    output_size=7,
    loss=QuantileLoss(),
    log_interval=2,
    reduce_on_plateau_patience=4
)
print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

# find optimal learning rate
res = trainer.lr_find(
    tft, train_dataloader=train_dataloader, val_dataloaders=val_dataloader, early_stop_threshold=1000.0, max_lr=0.3,
)

print(f"suggested learning rate: {res.suggestion()}")
fig = res.plot(show=True, suggest=True)
fig.show()

trainer.fit(
    tft, train_dataloader=train_dataloader, val_dataloaders=val_dataloader,
)
```

