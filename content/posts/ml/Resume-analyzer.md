---
title: "🚀 ResumeAnalyzer | An Easy Solution to Rank Resumes using Spacy"
date: 2020-10-10T00:42:27+05:30
draft: false
#featuredImage: "/posts/ml/images/resumeanalyzer.png"
featuredImagePreview: "/posts/ml/images/resumeanalyzer.png"
coverImage: "/posts/ml/images/resumeanalyzer.png"
images: ["/posts/ml/images/resumeanalyzer.png"]
tags: ["Resume Analyzer", "Machine Learning", "Spacy", "Python Packages"]
categories: ["Python Packages"]
description: "ResumeAnalyzer is an easy, lightweight python package to rank resumes based on your requirement in just one line of code."
---
<!--more-->

{{< admonition type=abstract title="Abstract" open=True >}}
🚀 **ResumeAnalyzer** is an easy, lightweight python package to rank resumes based on your requirement in just one line of code.
{{< /admonition >}}

{{< figure src="/posts/ml/images/resumeanalyzer.png" >}}






## Introduction
Let’s say, you have an opening for a data scientist in your organization. You post the requirement and you receive thousands of resumes. Great!

But there’s a challenge — how do you select the top 50 or 100 most relevant resumes against your requirement description for the next round?

If this problem sounds familiar to you, [ResumeAnalyzer](https://github.com/Shivanandroy/Resume-Analyzer) is here to rescue you.







>[ResumeAnalyzer](https://github.com/Shivanandroy/Resume-Analyzer) is an easy, lightweight python package to rank resumes based on your requirement in just one line of code.

## Demo

![Resume Analyzer](/posts/ml/images/resume-analyzer.gif)

<br>

## How it works?
It uses `textract` to process your documents, `spacy’s PhraseMatcher` to rank your resumes and `Dash` to render UI inside notebook as well as in browser. 

Here’s is a complete code looks like —

```python
# ! pip install ResumeAnalyzer
import ResumeAnalyzer as ra

analyzer = ra.ResumeAnalyzer()

# define the ranking criteria that suits your requirement
# E.g. rank candidates based on Deep Learning, Machine Learning and Time Series skills
search_criteria = {
    
    "Deep Learning": 
  ["neural networks", "cnn", "rnn", "ann", "lstm", "bert", "transformers"],
  
    "Machine Learning": 
  ["regression", "classification", "clustering", "time series", "summarization", "nlp"],
  
    "Time Series":  
  ["arima","sarimax", "prophet", "holt winters"]
  
}

# render in jupyter notebook
analyzer.render(path="Resume Folder/", metadata=search_criteria, mode="notebook")

# render in browser
analyzer.render(path="Resume Folder/", metadata=search_criteria, mode="browser")
```

## Let’s breakdown the code
#### 1 - Install, Import & Instantiate `ResumeAnalyzer`
```python
# install
! pip install ResumeAnalyzer

# import
import ResumeAnalyzer as ra

# instantiate
analyzer = ra.ResumeAnalyzer()

```

<br>

#### 2 - Define the rank criteria
A rank criteria is set of categories and its important terms to rank a candidate’s resume.

```python
# define the ranking criteria that suits your requirement
# E.g. rank candidates based on Deep Learning, Machine Learning and Time Series skills
search_criteria = {
    
    "Deep Learning": ["neural networks", "cnn", "rnn", "ann", "lstm", "bert", "transformers"],
  
    "Machine Learning": ["regression", "classification", "clustering", "time series", "summarization", "nlp"],
  
    "Time Series":  ["arima","sarimax", "prophet", "holt winters"]
  
}
```
<br>

`ResumeAnalyzer` passes this information to `Spacy’s PhraseMatcher` to calculate if that term is present in resume. If present, +1 is assigned to the resume in that category.

For e.g. — if *“Neural Network”* is present in the resume, the candidates is assigned a +1 in the **Deep Learning** category.

<br>

#### 3 - Render
`Render` renders the results inside jupyter notebook or browser using `Dash`.

```python
# render in jupyter notebook
analyzer.render(path="Resume Folder/", metadata=search_criteria, mode="notebook")

# render in browser
analyzer.render(path="Resume Folder/", metadata=search_criteria, mode="browser")
```
<br>

`render` takes 3 arguments —
 - **path** — path to resume folder
 - **metadata** — ranking criteria defined in step 2
 - **mode** — mode can be “notebook” if you want to visualize the results in Jupyter Notebook or “browser” to see the results in separate browser tab.

<br>

## 🥳Voila — We’re done!
{{< figure src="/posts/ml/images/RA.png" >}}

<br>

The Best Part !— The above table is **filterable** as well as **sortable**.

- You can sort candidates based on *“Deep Learning”* if you want to pickup candidates based on their deep learning skills
 - You can filter candidates with *“Total Score” > 3*
 - You can also select candidates based on percentile (Ranking): With *“Ranking” >0.7* means candidates above 70th percentile.


<br>

👋 **Try it out in Google Colab**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1UCNAhmVSKoXWS5E8Jg8iTMZbJBcxlrhb?usp=sharing)

<br>

## Notebooks
{{< admonition type=success title="Attachments" open=True >}}
- [Go to Github](https://github.com/Shivanandroy/Resume-Analyzer). 
- [Go to Google Colab Notebook](https://colab.research.google.com/drive/1UCNAhmVSKoXWS5E8Jg8iTMZbJBcxlrhb?usp=sharing)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1UCNAhmVSKoXWS5E8Jg8iTMZbJBcxlrhb?usp=sharing)
{{< /admonition >}}