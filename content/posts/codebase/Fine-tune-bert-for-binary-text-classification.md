---
title: "Code📝: Fine Tune BERT Model for Binary Text Classification"
date: 2020-10-08T00:42:27+05:30
draft: false
#featuredImage: "huggingface.png"
featuredImagePreview: "/posts/codebase/images/bert-binary-text-classification.png"
#coverImage: "huggingface.png"
images: ["/images/huggingface.png"]
tags: ["Deep Learning", "Transformers", "Text Classification","BERT"]
categories: ["Text Classification"]
description: "This is the code for downloading and fine tuning pre-trained BERT model on custom dataset for binary text classification"
---
<!--more-->



<iframe
  src="https://carbon.now.sh/embed?bg=rgba(255%2C255%2C255%2C1)&t=one-dark&wt=none&l=python&ds=true&dsyoff=20px&dsblur=68px&wc=true&wa=false&pv=56px&ph=56px&ln=false&fl=1&fm=Hack&fs=14px&lh=133%25&si=false&es=4x&wm=false&code=%2523%2520Requirements%253A%2520pytorch%253E%253D1.6%2520cudatoolkit%253D10.2%2520%250A%2523%2520install%2520it%2520from%2520here%253A%2520https%253A%252F%252Fpytorch.org%252F%250A%250A%2523%2520pip%2520install%2520simpletransformers%250Afrom%2520simpletransformers.classification%2520import%2520ClassificationModel%250Aimport%2520pandas%2520as%2520pd%250Aimport%2520logging%250A%250A%250Alogging.basicConfig(level%253Dlogging.INFO)%250Atransformers_logger%2520%253D%2520logging.getLogger(%2522transformers%2522)%250Atransformers_logger.setLevel(logging.WARNING)%250A%250A%2523%2520Train%2520and%2520Evaluation%2520data%2520needs%2520to%2520be%2520in%2520a%2520Pandas%2520Dataframe%2520of%2520two%2520columns.%2520The%2520first%2520column%2520is%2520the%2520text%2520with%2520type%2520str%252C%2520and%2520the%2520second%2520column%2520is%2520the%2520label%2520with%2520type%2520int.%250Atrain_data%2520%253D%2520%255B%255B%27Example%2520sentence%2520belonging%2520to%2520class%25201%27%252C%25201%255D%252C%2520%255B%27Example%2520sentence%2520belonging%2520to%2520class%25200%27%252C%25200%255D%255D%250Atrain_df%2520%253D%2520pd.DataFrame(train_data)%250A%250Aeval_data%2520%253D%2520%255B%255B%27Example%2520eval%2520sentence%2520belonging%2520to%2520class%25201%27%252C%25201%255D%252C%2520%255B%27Example%2520eval%2520sentence%2520belonging%2520to%2520class%25200%27%252C%25200%255D%255D%250Aeval_df%2520%253D%2520pd.DataFrame(eval_data)%250A%250A%2523%2520Create%2520a%2520ClassificationModel%2520(model_type%252C%2520model_name)%253A%2520For%2520e.g.%2520%250A%2523%2520(%27roberta%27%252C%2520%27roberta-base%27)%250A%2523%2520(%27albert%27%252C%27albert-base-v2%27)%250A%2523%2520(%27distilbert%27%252C%2520%27distilbert-base-uncased%27)%250A%2523%2520Supported%2520model%2520type%253A%2520CamemBERT%252C%2520RoBERTa%252C%2520DistilBERT%252C%2520ELECTRA%252C%2520FlauBERT%252C%2520Longformer%252C%2520MobileBERT%252C%2520XLM%252C%2520XLM-RoBERTa%252C%2520XLNet%250Amodel%2520%253D%2520ClassificationModel(%27bert%27%252C%2520%27bert-base-uncased%27)%2520%2523%2520You%2520can%2520set%2520class%2520weights%2520by%2520using%2520the%2520optional%2520weight%2520argument%250A%250A%2523%2520Train%2520the%2520model%250Amodel.train_model(train_df)%250A%250A%2523%2520Evaluate%2520the%2520model%250Aresult%252C%2520model_outputs%252C%2520wrong_predictions%2520%253D%2520model.eval_model(eval_df)"
  style="width: 800px; height: 1000px; border:0; transform: scale(1); overflow:hidden;"
  sandbox="allow-scripts allow-same-origin">
</iframe>

