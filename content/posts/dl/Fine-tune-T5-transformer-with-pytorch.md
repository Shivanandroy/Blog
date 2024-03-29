---
title: "Fine Tuning T5 Transformer Model with PyTorch"
date: 2021-02-08T00:42:27+05:30
draft: false
featuredImage: "/posts/dl/images/fine-tune-t5.png"
featuredImagePreview: "/posts/dl/images/fine-tune-t5.png"
coverImage: "/posts/dl/images/fine-tune-t5.png"
images: ["/posts/dl/images/fine-tune-t5.png"]
tags: ["Deep Learning", "Transformers", "T5"]
#categories: ["T5"]
description: "In this article, you will learn how to fine tune a T5 model with PyTorch and transformers"
---
<!--more-->

{{< admonition type=abstract title="Abstract" open=True >}}
In this article, you will learn how to fine tune a T5 transformer model using `PyTorch` & `Transformers🤗`

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1eoQUsisoPmc0e-bpjSKYYd-TE1F5YTqG?usp=sharing)
[![Go to GitHub Repository](https://img.shields.io/badge/Go%20to%20GitHub-Repository-green)](https://github.com/Shivanandroy/T5-Finetuning-PyTorch)
{{< /admonition >}}

## Introduction

A `T5` is an encoder-decoder model. It converts all NLP problems like language translation, summarization, text generation, question-answering, to a text-to-text task. 


For e.g., in case of **translation**, T5 accepts `source text`: English, as input and tries to convert it into `target text`: Serbian: 
| source text 	| target text 	|
|-	|-	|
| Hey, there! 	| Хеј тамо! 	|
| I'm going to train a T5 model with PyTorch 	| Обучићу модел Т5 са ПиТорцх-ом 	|


In case of **summarization**, source text or input can be a long description and target text can just be a one line summary. 
| source text                                                                                                                                                                                                                                                                                                                                                                               	| target text                                                       	|
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------	|-------------------------------------------------------------------	|
| "Saurav Kant, an alumnus of upGrad and IIIT-B's PG Program in Machine learning and Artificial Intelligence, was a Sr Systems Engineer at Infosys with almost 5 years of work experience. The program and upGrad's 360-degree career support helped him transition to a Data Scientist at Tech Mahindra with 90% salary hike. upGrad's Online Power Learning has powered 3 lakh+ careers." 	| upGrad learner switches to career in ML & Al with 90% salary hike 	|


In this article, we will take a pretrained `T5-base` model and fine tune it to generate a one line summary of news articles using `PyTorch`.


## Data
We will take a news summary dataset: It has 2 columns:
- `text` : article content
- `headlines` : one line summary of article content

```python
import pandas as pd

path = "https://raw.githubusercontent.com/Shivanandroy/T5-Finetuning-PyTorch/main/data/news_summary.csv"
df = pd.read_csv(path)

dh.head()

```
| text                                                                                                                                                                                                                                                                                                                                                                                                        | headlines                                                        |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| "Kunal Shah's credit card bill payment platform, CRED, gave users a chance to win free food from Swiggy for one year. Pranav Kaushik, a Delhi techie, bagged this reward after spending 2000 CRED coins. Users get one CRED coin per rupee of bill paid, which can be used to avail rewards from brands like Ixigo, BookMyShow, UberEats, Cult.Fit and more."                                               | Delhi techie wins free food from Swiggy for one year on CRED     |
| New Zealand defeated India by 8 wickets in the fourth ODI at Hamilton on Thursday to win their first match of the five-match ODI series. India lost an international match under Rohit Sharma's captaincy after 12 consecutive victories dating back to March 2018. The match witnessed India getting all out for 92, their seventh lowest total in ODI cricket history."                                   | New Zealand end Rohit Sharma-led India's 12-match winning streak |
| With Aegon Life iTerm Insurance plan, customers can enjoy tax benefits on your premiums paid and save up to â\x82¹46,800^ on taxes. The plan provides life cover up to the age of 100 years. Also, customers have options to insure against Critical Illnesses, Disability and Accidental Death Benefit Rider with a life cover up to the age of 80 years.'                                                 | Aegon life iTerm insurance plan helps customers save tax         |
| Isha Ghosh, an 81-year-old member of Bharat Scouts and Guides (BSG), has been imparting physical and mental training to schoolchildren in Jharkhand for several decades. Chaibasa-based Ghosh reportedly walks seven kilometres daily and spends eight hours conducting physical training, apart from climbing and yoga sessions. She says, "One should do something for society till one\'s last breath."' | 81-yr-old woman conducts physical training in J'khand schools    |


## Let's Code
`PyTorch` has a standard way to train any deep learning model. We will first start by writing a `Dataset` class, followed by `training`, `validation` steps and then a main `T5Trainer` function that will fine-tune our model.

But first let's install all the dependent modules and import them

### Import Libraries
```python
!pip install sentencepiece
!pip install transformers
!pip install torch
!pip install rich[jupyter]

# Importing libraries
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import os

# Importing the T5 modules from huggingface/transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration

# rich: for a better display on terminal
from rich.table import Column, Table
from rich import box
from rich.console import Console

# define a rich console logger
console = Console(record=True)

# to display dataframe in ASCII format
def display_df(df):
    """display dataframe in ASCII format"""

    console = Console()
    table = Table(
        Column("source_text", justify="center"),
        Column("target_text", justify="center"),
        title="Sample Data",
        pad_edge=False,
        box=box.ASCII,
    )

    for i, row in enumerate(df.values.tolist()):
        table.add_row(row[0], row[1])

    console.print(table)

# training logger to log training progress
training_logger = Table(
    Column("Epoch", justify="center"),
    Column("Steps", justify="center"),
    Column("Loss", justify="center"),
    title="Training Status",
    pad_edge=False,
    box=box.ASCII,
)

# Setting up the device for GPU usage
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

```
{{< advert1 >}}

### Dataset Class
We will write a `Dataset` class for reading our dataset and loading it into the dataloader and then feed it to the neural network for fine tuning the model.

This class will take 6 arguments as input:
- `dataframe (pandas.DataFrame)`: Input dataframe
- `tokenizer (transformers.tokenizer)`: T5 tokenizer
- `source_len (int)`: Max length of source text
- `target_len (int)`: Max length of target text
- `source_text (str)`: column name of source text
- `target_text (str)` : column name of target text

This class will have 2 methods: 
- `__len__`: returns the length of the dataframe
- `__getitem__`: return the input ids, attention masks and target ids

```python
class YourDataSetClass(Dataset):
    """
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the
    neural network for finetuning the model

    """

    def __init__(
        self, dataframe, tokenizer, source_len, target_len, source_text, target_text
    ):
        """
        Initializes a Dataset class

        Args:
            dataframe (pandas.DataFrame): Input dataframe
            tokenizer (transformers.tokenizer): Transformers tokenizer
            source_len (int): Max length of source text
            target_len (int): Max length of target text
            source_text (str): column name of source text
            target_text (str): column name of target text
        """
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = target_len
        self.target_text = self.data[target_text]
        self.source_text = self.data[source_text]

    def __len__(self):
        """returns the length of dataframe"""

        return len(self.target_text)

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""

        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])

        # cleaning data so as to ensure data is in string type
        source_text = " ".join(source_text.split())
        target_text = " ".join(target_text.split())

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.summ_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()
        target_mask = target["attention_mask"].squeeze()

        return {
            "source_ids": source_ids.to(dtype=torch.long),
            "source_mask": source_mask.to(dtype=torch.long),
            "target_ids": target_ids.to(dtype=torch.long),
            "target_ids_y": target_ids.to(dtype=torch.long),
        }
```

### Train steps

`train` function will the put model on training mode, generate outputs and calculate loss

This will take 6 arguments as input:
- `epoch`: epoch
- `tokenizer`: T5 tokenizer
- `model`: T5 model
- `loader`: Train Dataloader
- `optimizer`: Optimizer

```python
def train(epoch, tokenizer, model, device, loader, optimizer):

    """
    Function to be called for training with the parameters passed from main function

    """

    model.train()
    for _, data in enumerate(loader, 0):
        y = data["target_ids"].to(device, dtype=torch.long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data["source_ids"].to(device, dtype=torch.long)
        mask = data["source_mask"].to(device, dtype=torch.long)

        outputs = model(
            input_ids=ids,
            attention_mask=mask,
            decoder_input_ids=y_ids,
            labels=lm_labels,
        )
        loss = outputs[0]

        if _ % 10 == 0:
            training_logger.add_row(str(epoch), str(_), str(loss))
            console.print(training_logger)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

```
### Validation steps
`validate` function is same as the `train` function, but for the validation data

```python
def validate(epoch, tokenizer, model, device, loader):

  """
  Function to evaluate model for predictions

  """
  model.eval()
  predictions = []
  actuals = []
  with torch.no_grad():
      for _, data in enumerate(loader, 0):
          y = data['target_ids'].to(device, dtype = torch.long)
          ids = data['source_ids'].to(device, dtype = torch.long)
          mask = data['source_mask'].to(device, dtype = torch.long)

          generated_ids = model.generate(
              input_ids = ids,
              attention_mask = mask, 
              max_length=150, 
              num_beams=2,
              repetition_penalty=2.5, 
              length_penalty=1.0, 
              early_stopping=True
              )
          preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
          target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in y]
          if _%10==0:
              console.print(f'Completed {_}')

          predictions.extend(preds)
          actuals.extend(target)
  return predictions, actuals

```
### T5 Trainer

`T5Trainer` is our main function. It accepts input data, model type, model paramters to fine-tune the model. Under the hood, it utilizes, our `Dataset` class for data handling, `train` function to fine tune the model, `validate` to evaluate the model.

`T5Trainer` will have 5 arguments:
- `dataframe`: Input dataframe
- `source_text`: Column name of the input text i.e. article content
- `target_text`: Column name of the taregt text i.e. one line summary
- `model_params`: T5 model parameters
- `output_dir`: Output directory to save fine tuned T5 model.

```python
def T5Trainer(
    dataframe, source_text, target_text, model_params, output_dir="./outputs/"
):

    """
    T5 trainer

    """

    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(model_params["SEED"])  # pytorch random seed
    np.random.seed(model_params["SEED"])  # numpy random seed
    torch.backends.cudnn.deterministic = True

    # logging
    console.log(f"""[Model]: Loading {model_params["MODEL"]}...\n""")

    # tokenzier for encoding the text
    tokenizer = T5Tokenizer.from_pretrained(model_params["MODEL"])

    # Defining the model. We are using t5-base model and added a Language model layer on top for generation of Summary.
    # Further this model is sent to device (GPU/TPU) for using the hardware.
    model = T5ForConditionalGeneration.from_pretrained(model_params["MODEL"])
    model = model.to(device)

    # logging
    console.log(f"[Data]: Reading data...\n")

    # Importing the raw dataset
    dataframe = dataframe[[source_text, target_text]]
    display_df(dataframe.head(2))

    # Creation of Dataset and Dataloader
    # Defining the train size. So 80% of the data will be used for training and the rest for validation.
    train_size = 0.8
    train_dataset = dataframe.sample(frac=train_size, random_state=model_params["SEED"])
    val_dataset = dataframe.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)

    console.print(f"FULL Dataset: {dataframe.shape}")
    console.print(f"TRAIN Dataset: {train_dataset.shape}")
    console.print(f"TEST Dataset: {val_dataset.shape}\n")

    # Creating the Training and Validation dataset for further creation of Dataloader
    training_set = YourDataSetClass(
        train_dataset,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        source_text,
        target_text,
    )
    val_set = YourDataSetClass(
        val_dataset,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        source_text,
        target_text,
    )

    # Defining the parameters for creation of dataloaders
    train_params = {
        "batch_size": model_params["TRAIN_BATCH_SIZE"],
        "shuffle": True,
        "num_workers": 0,
    }

    val_params = {
        "batch_size": model_params["VALID_BATCH_SIZE"],
        "shuffle": False,
        "num_workers": 0,
    }

    # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
    training_loader = DataLoader(training_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)

    # Defining the optimizer that will be used to tune the weights of the network in the training session.
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=model_params["LEARNING_RATE"]
    )

    # Training loop
    console.log(f"[Initiating Fine Tuning]...\n")

    for epoch in range(model_params["TRAIN_EPOCHS"]):
        train(epoch, tokenizer, model, device, training_loader, optimizer)

    console.log(f"[Saving Model]...\n")
    # Saving the model after training
    path = os.path.join(output_dir, "model_files")
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)

    # evaluating test dataset
    console.log(f"[Initiating Validation]...\n")
    for epoch in range(model_params["VAL_EPOCHS"]):
        predictions, actuals = validate(epoch, tokenizer, model, device, val_loader)
        final_df = pd.DataFrame({"Generated Text": predictions, "Actual Text": actuals})
        final_df.to_csv(os.path.join(output_dir, "predictions.csv"))

    console.save_text(os.path.join(output_dir, "logs.txt"))

    console.log(f"[Validation Completed.]\n")
    console.print(
        f"""[Model] Model saved @ {os.path.join(output_dir, "model_files")}\n"""
    )
    console.print(
        f"""[Validation] Generation on Validation data saved @ {os.path.join(output_dir,'predictions.csv')}\n"""
    )
    console.print(f"""[Logs] Logs saved @ {os.path.join(output_dir,'logs.txt')}\n""")
```

### Model Parameters
`model_params` is a dictionary containing model paramters for T5 training:

- `MODEL: "t5-base"`,  model_type: t5-base/t5-large
- `TRAIN_BATCH_SIZE: 8`,  training batch size
- `VALID_BATCH_SIZE: 8`,  validation batch size
- `TRAIN_EPOCHS:3`,  number of training epochs
- `VAL_EPOCHS: 1`,  number of validation epochs
- `LEARNING_RATE: 1e-4`,  learning rate
- `MAX_SOURCE_TEXT_LENGTH: 512`,  max length of source text
- `MAX_TARGET_TEXT_LENGTH: 50`,   max length of target text
- `SEED: 42`,  set seed for reproducibility

```python

# let's define model parameters specific to T5
model_params = {
    "MODEL": "t5-base",  # model_type: t5-base/t5-large
    "TRAIN_BATCH_SIZE": 8,  # training batch size
    "VALID_BATCH_SIZE": 8,  # validation batch size
    "TRAIN_EPOCHS": 3,  # number of training epochs
    "VAL_EPOCHS": 1,  # number of validation epochs
    "LEARNING_RATE": 1e-4,  # learning rate
    "MAX_SOURCE_TEXT_LENGTH": 512,  # max length of source text
    "MAX_TARGET_TEXT_LENGTH": 50,  # max length of target text
    "SEED": 42,  # set seed for reproducibility
}
```

### Let's call T5Trainer

```python
# T5 accepts prefix of the task to be performed:
# Since we are summarizing, let's add summarize to source text as a prefix
df["text"] = "summarize: " + df["text"]

T5Trainer(
    dataframe=df,
    source_text="text",
    target_text="headlines",
    model_params=model_params,
    output_dir="outputs",
)

```
```ASCII
                              Training Status                               
+--------------------------------------------------------------------------+
|Epoch | Steps |                            Loss                           |
|------+-------+-----------------------------------------------------------|
|  0   |   0   | tensor(8.5338, device='cuda:0', grad_fn=<NllLossBackward>)|
|  0   |  10   | tensor(3.4278, device='cuda:0', grad_fn=<NllLossBackward>)|
|  0   |  20   | tensor(3.0148, device='cuda:0', grad_fn=<NllLossBackward>)|
|  0   |  30   | tensor(3.2338, device='cuda:0', grad_fn=<NllLossBackward>)|
|  0   |  40   | tensor(2.5963, device='cuda:0', grad_fn=<NllLossBackward>)|
|  1   |   0   | tensor(2.2411, device='cuda:0', grad_fn=<NllLossBackward>)|
|  1   |  10   | tensor(1.9470, device='cuda:0', grad_fn=<NllLossBackward>)|
|  1   |  20   | tensor(1.9091, device='cuda:0', grad_fn=<NllLossBackward>)|
|  1   |  30   | tensor(2.0122, device='cuda:0', grad_fn=<NllLossBackward>)|
|  1   |  40   | tensor(1.5261, device='cuda:0', grad_fn=<NllLossBackward>)|
|  2   |   0   | tensor(1.6496, device='cuda:0', grad_fn=<NllLossBackward>)|
|  2   |  10   | tensor(1.1971, device='cuda:0', grad_fn=<NllLossBackward>)|
|  2   |  20   | tensor(1.6908, device='cuda:0', grad_fn=<NllLossBackward>)|
|  2   |  30   | tensor(1.4069, device='cuda:0', grad_fn=<NllLossBackward>)|
|  2   |  40   | tensor(2.1261, device='cuda:0', grad_fn=<NllLossBackward>)|
+--------------------------------------------------------------------------+
```

## Notebooks
{{< admonition type=success title="Attachments" open=True >}}
- [Go to Dataset](https://raw.githubusercontent.com/Shivanandroy/T5-Finetuning-PyTorch/main/data/news_summary.csv). 
- [Go to Notebook](https://github.com/Shivanandroy/T5-Finetuning-PyTorch/blob/main/notebook/T5_Fine_tuning_with_PyTorch.ipynb)
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1eoQUsisoPmc0e-bpjSKYYd-TE1F5YTqG?usp=sharing)
- [![Go to GitHub Repository](https://img.shields.io/badge/Go%20to%20GitHub-Repository-green)](https://github.com/Shivanandroy/T5-Finetuning-PyTorch)
{{< /admonition >}}