# Code📝: Fine Tune BERT Model for Binary Text Classification

<!--more-->

```python
# Requirements: pytorch>=1.6 cudatoolkit=10.2 
# install it from here: https://pytorch.org/

# pip install simpletransformers
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

# Create a ClassificationModel (model_type, model_name): For e.g. 
# ('roberta', 'roberta-base')
# ('albert','albert-base-v2')
# ('distilbert', 'distilbert-base-uncased')
# Supported model type: CamemBERT, RoBERTa, DistilBERT, ELECTRA, FlauBERT, Longformer, MobileBERT, XLM, XLM-RoBERTa, XLNet
model = ClassificationModel('bert', 'bert-base-uncased') # You can set class weights by using the optional weight argument

# Train the model
model.train_model(train_df)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(eval_df)
```




