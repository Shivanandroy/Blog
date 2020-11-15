---
title: "Visualizing Game of Thrones with BERT"
date: 2020-11-13T00:42:27+05:30
draft: false
#featuredImage: "huggingface.png"
featuredImagePreview: "/posts/dl/images/GOT.png"
#coverImage: "/posts/dl/images/covid19.jpg"
#images: ["/posts/dl/images/covid19.jpg"]
tags: ["Deep Learning", "Game of Thrones", "BERT"]
categories: ["Natural Language Understanding"]
description: "In this article, we will visualize Game of Thrones books with BERT in 3D space."

---
<!--more-->
{{< admonition type=abstract title="Abstract" open=True >}}
In this article, we will visualize **Game of Thrones** books with **BERT** in 3D embedding space .
{{< /admonition >}}

{{< figure src="/posts/dl/images/GOT.png" >}}


<br>

## Introduction

This past weekend while watching Game of Thrones at dinner — I had a thought!

**How does BERT understand Game of Thrones?**

The thought of visualizing all the texts of GOT books with mightly BERT in 3D space.

How can we achieve this —
- First — we will extract the BERT embeddings for each word across all GOT books.
- Then — reduce the dimension of BERT embeddings to visualize it in 3D
- And finally — create a web application to visualize it on the browser

<br>

 {{< figure src="/posts/dl/images/got.gif" >}}

<br>
So, Let's get started
<br>

## 1. Extracting BERT Embeddings for Game of Thrones Books

Extracting BERT embeddings for your custom data can be intimidating at first — but not anymore. 

[Gary Lai](https://github.com/imgarylai) has this awesome package [`bert-embedding`](https://github.com/imgarylai/bert-embedding) which lets you extract token level embeddings without any hassle. The code looks as simple as:

```python
# installing bert-embedding
!pip install bert-embedding
# importing bert_embedding
from bert_embedding import BertEmbedding
# text to be encoded
text = """We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers.
 Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations by jointly conditioning on both left and right context in all layers.
"""
# generating sentences
sentences = text.split('\n')
# instantiating BerEmbedding class
bert_embedding = BertEmbedding()
# passing sentences to bert_embedding model
result = bert_embedding(sentences)
```
<br>

Let’s use this package for our data —
We will extract 5 Game of Thrones books using `requests` —

```python
import requests

book1 = "https://raw.githubusercontent.com/llSourcell/word_vectors_game_of_thrones-LIVE/master/data/got1.txt"
book2 = "https://raw.githubusercontent.com/llSourcell/word_vectors_game_of_thrones-LIVE/master/data/got2.txt"
book3 = "https://raw.githubusercontent.com/llSourcell/word_vectors_game_of_thrones-LIVE/master/data/got3.txt"
book4 = "https://raw.githubusercontent.com/llSourcell/word_vectors_game_of_thrones-LIVE/master/data/got4.txt"
book5 = "https://raw.githubusercontent.com/llSourcell/word_vectors_game_of_thrones-LIVE/master/data/got5.txt"

# fetch the content of the books
b1 = requests.get(book1)
b2 = requests.get(book2)
b3 = requests.get(book3)
b4 = requests.get(book4)
b5 = requests.get(book5)

# break it into list of sentences
book1_content = [sent for sent in b1.text.splitlines() if sent != '']
book2_content = [sent for sent in b2.text.splitlines() if sent != '']
book3_content = [sent for sent in b3.text.splitlines() if sent != '']
book4_content = [sent for sent in b4.text.splitlines() if sent != '']
book5_content = [sent for sent in b5.text.splitlines() if sent != '']
```
<br>
Next — We will clean the content of the books. And we will store the content as a list of sentences

```python
import re
def sentence_to_wordlist(raw):
    clean = re.sub(“[^a-zA-Z]”,” “, raw)
    words = clean.split()
    return words

book1_sentences = []
for raw_sentence in book1_content:
    if len(raw_sentence) > 0:
        book1_sentences.append(' '.join(sentence_to_wordlist(raw_sentence)))

book2_sentences = []
for raw_sentence in book2_content:
    if len(raw_sentence) > 0:
        book2_sentences.append(' '.join(sentence_to_wordlist(raw_sentence)))

book3_sentences = []
for raw_sentence in book3_content:
    if len(raw_sentence) > 0:
        book3_sentences.append(' '.join(sentence_to_wordlist(raw_sentence)))

book4_sentences = []
for raw_sentence in book4_content:
    if len(raw_sentence) > 0:
        book4_sentences.append(' '.join(sentence_to_wordlist(raw_sentence)))

book5_sentences = []
for raw_sentence in book5_content:
    if len(raw_sentence) > 0:
        book5_sentences.append(' '.join(sentence_to_wordlist(raw_sentence)))
```
<br>
Once we have a clean list of sentences for each book, we can extract BERT embeddings using the code below:

```python
# imorting dependencies
from bert_embedding import BertEmbedding
from tqdm import tqdm_notebook
import pandas as pd
import mxnet as mx
# bert_embedding supports GPU for faster processsing
ctx = mx.gpu(0)
# This function will extract BERT embeddings and store it in a 
# structured format i.e. dataframe
def generate_bert_embeddings(sentences):
    bert_embedding = BertEmbedding(ctx=ctx)
    print(“Encoding Sentences:”)
    result = bert_embedding(sentences)
    print(“Encoding Finished”)
    df = pd.DataFrame()
    for i in tqdm_notebook(range(len(result))):
        embed = pd.DataFrame(result[i][1])
        embed[‘words’] = result[i][0]
        df = pd.concat([df, embed])
    return df
book1_embedding = generate_bert_embeddings(book1_sentences)
book2_embedding = generate_bert_embeddings(book2_sentences)
book3_embedding = generate_bert_embeddings(book3_sentences)
book4_embedding = generate_bert_embeddings(book4_sentences)
book5_embedding = generate_bert_embeddings(book5_sentences)
```

><b><i>These embeddings are out of pre-trained BERT model. You can also fine-tune BERT on GOT texts before fetching the embeddings.</i></b>

<br>

## 2. Dimensionality Reduction: BERT Embeddings
BERT embeddings are 768 dimension vectors i.e. we have 768 numbers to represent each word or tokens found in the books.

We will reduce the dimensionality of these words from 768 to 3 — to visualize these tokens/words in 3 dimensions using the code below:

```python
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
# This function will reduce dimension of the embeddings using tSVD 
# and PCA
def reduce_dimension(embedding_df):
# Dimensionality Reduction using tSVD
    tsvd = TruncatedSVD(n_components=3)
    tsvd_3d = pd.DataFrame(tsvd.fit_transform(embedding_df.drop(‘words’, axis=1)))
    tsvd_3d[‘words’] = embedding_df[‘words’].values

# Dimensionality reduction using PCA
    pca = PCA(3)
    pca_3d = pd.DataFrame(pca.fit_transform(embedding_df.drop(‘words’, axis=1)))
    pca_3d[‘words’] = embedding_df[‘words’].values
    return tsvd_3d, pca_3d
```
<br>
Let’s apply the above function to our embeddings

```python
tsvd_book1, pca_book1 = reduce_dimension(book1_embedding)
tsvd_book2, pca_book2 = reduce_dimension(book2_embedding)
tsvd_book3, pca_book3 = reduce_dimension(book3_embedding)
tsvd_book4, pca_book4 = reduce_dimension(book4_embedding)
tsvd_book5, pca_book5 = reduce_dimension(book5_embedding)
```
<br>

🥳 Voila! Now we have 3 dimension projection of each word in all the GOT books

>Extraction of BERT embeddings and dimensionality reduction can be a time-consuming process. You can download Game of Thrones BERT Embeddings from here: [Download](https://drive.google.com/open?id=1M5vHLQqCv_AB1dm9kXW4AHsA5CjcFrm1)

<br>

## 3. Building A Web App to visualize on the Browser
This is the final part of this project. We will build a front end to visualize these embeddings in 3 dimensions in pure python.

To do this, we will use `Dash`. 

`Dash` is a python framework that lets you build beautiful web-based analytical apps in pure python. No JavaScript required.

You can install dash : `pip install dash`

> ***If you need a end-to-end article on how to build web apps in pure python with `Dash`, let me know in the comments***

A Dash application consists of 3parts —

**1. Dependencies and app instantiation**

This section talks about importing dependent packages and starting a Dash app

```python
# pip install dash==1.8.0
import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_table
from dash.dependencies import Input, Output, State

# pip install dash_bootstrap_components
import dash_bootstrap_components as dbc

# pip install plotly_express
import plotly_express as px

# pip install sd_material_ui
import sd_material_ui as sd

# instantiating dash application
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP,"https://codepen.io/chriddyp/pen/brPBPO.css"])

# adding a title to your dash app
app.title="Visualizing Game of Thrones Using BERT"
```
<br>

**2. Layout**

It lets you define how your web application would look like - widgets, sliders, dropdowns etc. & their alignment

```python
# Loading the data: You can download it from 
# https://drive.google.com/open?id=1M5vHLQqCv_AB1dm9kXW4AHsA5CjcFrm1
data = pd.read_csv("got_embeddings.csv")

# Defining the app layout
app.layout = html.Div([

	html.Div(html.H4("Visualizing Game of Thrones with BERT"),
			style={'textAlign':'center','backgroundColor':'#ff8533','color':'white','font-size':5,'padding':'1px'}),

	dbc.Row([
		dbc.Col([

			html.Br(),
			html.Div("Choose a Book", style={'font-weight':'bold'}),
			sd.DropDownMenu(id='book',
                                    value='Book 1',
                                    options=[
                                        dict(value='Book 1', primaryText='Game of Thrones 1',
                                             label='Game of Thrones 1'),
                                        dict(value='Book 2', primaryText='Game of Thrones 2'),
                                        dict(value='Book 3', primaryText='Game of Thrones 3'),
										dict(value='Book 4', primaryText='Game of Thrones 4'),
                                        dict(value='Book 5', primaryText='Game of Thrones 5'),
                                    ],
                                    menuStyle=dict(width=300),  # controls style of the open menu
                                    listStyle=dict(height=35),
                                    selectedMenuItemStyle=dict(height=30),
                                    anchorOrigin=dict(vertical='bottom', horizontal='right')),


			html.Hr(),

			html.Div("Number of words:", style={'font-weight':'bold'}),
			dcc.Slider(id='num_words', min=0, tooltip={'always_visible':False}, value=5000, max=10000, step=100),

			html.Hr(),

			html.Div("Projection", style={'font-weight':'bold'}),
			dbc.RadioItems( options=[{"label": "Truncated SVD", "value": "tSVD"},
									 {"label": "PCA", "value": "PCA"}],
									 value="tSVD", id="projection"),

			html.Hr(),

			html.Div("Options", style={'font-weight':'bold'}),
			dbc.Checklist(options=[{"label": "Show Noun Phrases", "value": 'noun'}],
			            	id="noun_toggle",switch=True, value=[]),
			dbc.Checklist(options=[{"label": "Show Unique Words", "value": 'unique'}],
			            	id="unique_toggle",switch=True, value=['unique']),
			dbc.Checklist(options=[{"label": "Remove Stopwords", "value": 'stopword'}],
			            	id="stopword_toggle",switch=True, value=['stopword']),

			html.Hr(),

		], width=2),

		dbc.Col(dcc.Graph(id='visualization'), width=10)




	], no_gutters=True)

	])
```

<br>

**3. Callbacks**

It lets you add interactivity on your charts, visuals or buttons.

```python
@app.callback(Output("visualization", "figure"),
    [
        Input("book", "value"),
        Input("num_words", "value"),
        Input("projection", "value"),
		Input("noun_toggle", "value"),
		Input("unique_toggle", "value"),
		Input("stopword_toggle", "value")
    ],
)
def on_form_change(book_num, num_words, projection, is_noun, is_unique, is_stopwords):

	df = data[(data.book == book_num) & (data.type == projection) & (data.length != 1)]
	df['word_usage'] = pd.qcut(df.frequency,5, labels=['Rare','Less Frequent','Moderate', 'Frequent','Most Frequent'])
	if "noun" in is_noun:
		df = df[df.pos == "NN"]
	if "unique" in is_unique:
		df = df.loc[df.words.drop_duplicates().index]

	if "stopword" in is_stopwords:
		df = df[df.stopwords == False]
	df = df.sort_values(by='frequency', ascending=False)[:num_words]
	n_words = df.shape[0]
	fig = px.scatter_3d(df, x='Dimension 1', y="Dimension 2", z="Dimension 3", height=600, size='frequency', color='frequency', size_max=40,hover_name='words')

	fig.update_layout(
			scene = dict(
                    xaxis = dict(
                         backgroundcolor="white",
                         gridcolor="white",
                         showbackground=True,
                         zerolinecolor="white",
						 ticks='',
						showticklabels=False),
                    yaxis = dict(
                        backgroundcolor="white",
                        gridcolor="white",
                        showbackground=True,
                        zerolinecolor="white",
						ticks='',
						showticklabels=False),
                    zaxis = dict(
                        backgroundcolor="white",
                        gridcolor="white",
                        showbackground=True,
                        zerolinecolor="white",
						ticks='',
						showticklabels=False),
					xaxis_title='.',
                    yaxis_title='.',
                    zaxis_title='.'

						),


                  )
	return fig



if __name__ == '__main__':
    app.run_server(host='0.0.0.0',debug=True, port=8050)
```

<br>

Lets write all the 3 parts of the Dash app in a single `app.py` file and run the `app.py` file in your terminal: 

`>> python app.py`

```python
# importing packages
import pandas as pd

# pip install dash==1.8.0
import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_table
from dash.dependencies import Input, Output, State

# pip install dash_bootstrap_components
import dash_bootstrap_components as dbc

# pip install plotly_express
import plotly_express as px

# pip install sd_material_ui
import sd_material_ui as sd

# instantiating dash application
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP,"https://codepen.io/chriddyp/pen/brPBPO.css"])
# adding a title to your dash app
app.title="Visualizing Game of Thrones Using BERT"
# Loading screen CSS
#app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/brPBPO.css"})

data = pd.read_csv("got_embeddings.csv")

#fig = px.scatter_3d(df[:10000], x='Dimension 1', y="Dimension 2", z="Dimension 3", height=600, size='frequency', color='frequency', hover_name='words')

app.layout = html.Div([

	html.Div(html.H4("Visualizing Game of Thrones with BERT"),
			style={'textAlign':'center','backgroundColor':'#ff8533','color':'white','font-size':5,'padding':'1px'}),

	dbc.Row([
		dbc.Col([

			html.Br(),
			html.Div("Choose a Book", style={'font-weight':'bold'}),
			sd.DropDownMenu(id='book',
                                    value='Book 1',
                                    options=[
                                        dict(value='Book 1', primaryText='Game of Thrones 1',
                                             label='Game of Thrones 1'),
                                        dict(value='Book 2', primaryText='Game of Thrones 2'),
                                        dict(value='Book 3', primaryText='Game of Thrones 3'),
										dict(value='Book 4', primaryText='Game of Thrones 4'),
                                        dict(value='Book 5', primaryText='Game of Thrones 5'),
                                    ],
                                    menuStyle=dict(width=300),  # controls style of the open menu
                                    listStyle=dict(height=35),
                                    selectedMenuItemStyle=dict(height=30),
                                    anchorOrigin=dict(vertical='bottom', horizontal='right')),


			html.Hr(),

			html.Div("Number of words:", style={'font-weight':'bold'}),
			dcc.Slider(id='num_words', min=0, tooltip={'always_visible':False}, value=5000, max=10000, step=100),

			html.Hr(),

			html.Div("Projection", style={'font-weight':'bold'}),
			dbc.RadioItems( options=[{"label": "Truncated SVD", "value": "tSVD"},
									 {"label": "PCA", "value": "PCA"}],
									 value="tSVD", id="projection"),

			html.Hr(),

			html.Div("Options", style={'font-weight':'bold'}),
			dbc.Checklist(options=[{"label": "Show Noun Phrases", "value": 'noun'}],
			            	id="noun_toggle",switch=True, value=[]),
			dbc.Checklist(options=[{"label": "Show Unique Words", "value": 'unique'}],
			            	id="unique_toggle",switch=True, value=['unique']),
			dbc.Checklist(options=[{"label": "Remove Stopwords", "value": 'stopword'}],
			            	id="stopword_toggle",switch=True, value=['stopword']),

			html.Hr(),

		], width=2),

		dbc.Col(dcc.Graph(id='visualization'), width=10)




	], no_gutters=True)

	])



@app.callback(Output("visualization", "figure"),
    [
        Input("book", "value"),
        Input("num_words", "value"),
        Input("projection", "value"),
		Input("noun_toggle", "value"),
		Input("unique_toggle", "value"),
		Input("stopword_toggle", "value")
    ],
)
def on_form_change(book_num, num_words, projection, is_noun, is_unique, is_stopwords):

	df = data[(data.book == book_num) & (data.type == projection) & (data.length != 1)]
	df['word_usage'] = pd.qcut(df.frequency,5, labels=['Rare','Less Frequent','Moderate', 'Frequent','Most Frequent'])
	if "noun" in is_noun:
		df = df[df.pos == "NN"]
	if "unique" in is_unique:
		df = df.loc[df.words.drop_duplicates().index]

	if "stopword" in is_stopwords:
		df = df[df.stopwords == False]
	df = df.sort_values(by='frequency', ascending=False)[:num_words]
	n_words = df.shape[0]
	fig = px.scatter_3d(df, x='Dimension 1', y="Dimension 2", z="Dimension 3", height=600, size='frequency', color='frequency', size_max=40,hover_name='words')

	fig.update_layout(
			scene = dict(
                    xaxis = dict(
                         backgroundcolor="white",
                         gridcolor="white",
                         showbackground=True,
                         zerolinecolor="white",
						 ticks='',
						showticklabels=False),
                    yaxis = dict(
                        backgroundcolor="white",
                        gridcolor="white",
                        showbackground=True,
                        zerolinecolor="white",
						ticks='',
						showticklabels=False),
                    zaxis = dict(
                        backgroundcolor="white",
                        gridcolor="white",
                        showbackground=True,
                        zerolinecolor="white",
						ticks='',
						showticklabels=False),
					xaxis_title='.',
                    yaxis_title='.',
                    zaxis_title='.'

						),


                  )
	return fig



if __name__ == '__main__':
    app.run_server(host='0.0.0.0',debug=True, port=8050)
```

<br>

##  🥳 Hooray, you’re done!

Now you can explore your GOT characters in 3D.

 {{< figure src="/posts/dl/images/got.gif" >}}

<br>

> **But, what did I find out from this experiment?**
>- Were all characters, food items, places, things formed seperate clusters?
> - Which all characters were in close proximity?

**I will keep it for the next time**


