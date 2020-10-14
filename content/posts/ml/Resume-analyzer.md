---
title: "🚀 ResumeAnalyzer | An Easy Solution to Rank Resumes using Spacy"
date: 2020-10-10T00:42:27+05:30
draft: false
#featuredImage: "huggingface.png"
featuredImagePreview: "/posts/ml/images/resumeanalyzer.png"
#coverImage: "huggingface.png"
#images: ["/images/huggingface.png"]
tags: ["Resume Analyzer", "Machine Learning", "Spacy"]
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

<iframe
  src="https://carbon.now.sh/embed?bg=rgba(255%2C255%2C255%2C1)&t=one-dark&wt=none&l=python&ds=true&dsyoff=20px&dsblur=68px&wc=true&wa=false&pv=56px&ph=56px&ln=false&fl=1&fm=Hack&fs=14px&lh=133%25&si=false&es=4x&wm=false&code=%2523%2520!%2520pip%2520install%2520ResumeAnalyzer%250Aimport%2520ResumeAnalyzer%2520as%2520ra%250A%250Aanalyzer%2520%253D%2520ra.ResumeAnalyzer()%250A%250A%2523%2520define%2520the%2520ranking%2520criteria%2520that%2520suits%2520your%2520requirement%250A%2523%2520E.g.%2520rank%2520candidates%2520based%2520on%2520Deep%2520Learning%252C%2520Machine%2520Learning%2520and%2520Time%2520Series%2520skills%250Asearch_criteria%2520%253D%2520%257B%250A%2520%2520%2520%2520%250A%2520%2520%2520%2520%2522Deep%2520Learning%2522%253A%2520%250A%2520%2520%255B%2522neural%2520networks%2522%252C%2520%2522cnn%2522%252C%2520%2522rnn%2522%252C%2520%2522ann%2522%252C%2520%2522lstm%2522%252C%2520%2522bert%2522%252C%2520%2522transformers%2522%255D%252C%250A%2520%2520%250A%2520%2520%2520%2520%2522Machine%2520Learning%2522%253A%2520%250A%2520%2520%255B%2522regression%2522%252C%2520%2522classification%2522%252C%2520%2522clustering%2522%252C%2520%2522time%2520series%2522%252C%2520%2522summarization%2522%252C%2520%2522nlp%2522%255D%252C%250A%2520%2520%250A%2520%2520%2520%2520%2522Time%2520Series%2522%253A%2520%2520%250A%2520%2520%255B%2522arima%2522%252C%2522sarimax%2522%252C%2520%2522prophet%2522%252C%2520%2522holt%2520winters%2522%255D%250A%2520%2520%250A%257D%250A%250A%2523%2520render%2520in%2520jupyter%2520notebook%250Aanalyzer.render(path%253D%2522Resume%2520Folder%252F%2522%252C%2520metadata%253Dsearch_criteria%252C%2520mode%253D%2522notebook%2522)%250A%250A%2523%2520render%2520in%2520browser%250Aanalyzer.render(path%253D%2522Resume%2520Folder%252F%2522%252C%2520metadata%253Dsearch_criteria%252C%2520mode%253D%2522browser%2522)"
  style="width: 800px; height: 800px; border:0; transform: scale(1); overflow:hidden;"
  sandbox="allow-scripts allow-same-origin">
</iframe>

## Let’s breakdown the code
#### 1 - Install, Import & Instantiate `ResumeAnalyzer`
<iframe
  src="https://carbon.now.sh/embed?bg=rgba(255%2C255%2C255%2C1)&t=one-dark&wt=none&l=python&ds=true&dsyoff=20px&dsblur=68px&wc=true&wa=false&pv=56px&ph=56px&ln=false&fl=1&fm=Hack&fs=14px&lh=133%25&si=false&es=4x&wm=false&code=%2523%2520install%250A!%2520pip%2520install%2520ResumeAnalyzer%250A%250A%2523%2520import%250Aimport%2520ResumeAnalyzer%2520as%2520ra%250A%250A%2523%2520instantiate%250Aanalyzer%2520%253D%2520ra.ResumeAnalyzer()%250A"
  style="width: 800px; height: 352px; border:0; transform: scale(1); overflow:hidden;"
  sandbox="allow-scripts allow-same-origin">
</iframe>

<br>

#### 2 - Define the rank criteria
A rank criteria is set of categories and its important terms to rank a candidate’s resume.

<iframe
  src="https://carbon.now.sh/embed?bg=rgba(255%2C255%2C255%2C1)&t=one-dark&wt=none&l=python&ds=true&dsyoff=20px&dsblur=68px&wc=true&wa=false&pv=56px&ph=56px&ln=false&fl=1&fm=Hack&fs=14px&lh=133%25&si=false&es=4x&wm=false&code=%2523%2520define%2520the%2520ranking%2520criteria%2520that%2520suits%2520your%2520requirement%250A%2523%2520E.g.%2520rank%2520candidates%2520based%2520on%2520Deep%2520Learning%252C%2520Machine%2520Learning%2520and%2520Time%2520Series%2520skills%250Asearch_criteria%2520%253D%2520%257B%250A%2520%2520%2520%2520%250A%2520%2520%2520%2520%2522Deep%2520Learning%2522%253A%2520%250A%2520%2520%255B%2522neural%2520networks%2522%252C%2520%2522cnn%2522%252C%2520%2522rnn%2522%252C%2520%2522ann%2522%252C%2520%2522lstm%2522%252C%2520%2522bert%2522%252C%2520%2522transformers%2522%255D%252C%250A%2520%2520%250A%2520%2520%2520%2520%2522Machine%2520Learning%2522%253A%2520%250A%2520%2520%255B%2522regression%2522%252C%2520%2522classification%2522%252C%2520%2522clustering%2522%252C%2520%2522time%2520series%2522%252C%2520%2522summarization%2522%252C%2520%2522nlp%2522%255D%252C%250A%2520%2520%250A%2520%2520%2520%2520%2522Time%2520Series%2522%253A%2520%2520%250A%2520%2520%255B%2522arima%2522%252C%2522sarimax%2522%252C%2520%2522prophet%2522%252C%2520%2522holt%2520winters%2522%255D%250A%2520%2520%250A%257D"
  style="width: 800px; height: 500px; border:0; transform: scale(1); overflow:hidden;"
  sandbox="allow-scripts allow-same-origin">
</iframe>

`ResumeAnalyzer` passes this information to `Spacy’s PhraseMatcher` to calculate if that term is present in resume. If present, +1 is assigned to the resume in that category.

For e.g. — if *“Neural Network”* is present in the resume, the candidates is assigned a +1 in the **Deep Learning** category.

<br>

#### 3 - Render
`Render` renders the results inside jupyter notebook or browser using `Dash`.
<iframe
  src="https://carbon.now.sh/embed?bg=rgba(255%2C255%2C255%2C1)&t=one-dark&wt=none&l=python&ds=true&dsyoff=20px&dsblur=68px&wc=true&wa=false&pv=56px&ph=56px&ln=false&fl=1&fm=Hack&fs=14px&lh=133%25&si=false&es=4x&wm=false&code=%2523%2520render%2520in%2520jupyter%2520notebook%250Aanalyzer.render(path%253D%2522Resume%2520Folder%252F%2522%252C%2520metadata%253Dsearch_criteria%252C%2520mode%253D%2522notebook%2522)%250A%250A%2523%2520render%2520in%2520browser%250Aanalyzer.render(path%253D%2522Resume%2520Folder%252F%2522%252C%2520metadata%253Dsearch_criteria%252C%2520mode%253D%2522browser%2522)"
  style="width: 800px; height: 315px; border:0; transform: scale(1); overflow:hidden;"
  sandbox="allow-scripts allow-same-origin">
</iframe>


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