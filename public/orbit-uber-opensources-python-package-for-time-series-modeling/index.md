# Orbit | Uber's New Open Source Python Library for Time Series Modeling

<!--more-->


`Orbit` is Uber's new python package for time series modeling and inference using Bayesian sampling methods for model estimation. 

{{< figure src="/posts/ml/images/orbit.png" >}}

**Orbit** provides a familiar and intuitive initialize-fit-predict interface for working with time series tasks, while utilizing probabilistic modeling under the hood.



As per Orbit's documentation, initial release supports concrete implementation for the following models:

- Local Global Trend (LGT)
- Damped Local Trend (DLT)



Both models, which are variants of exponential smoothing, support seasonality and exogenous (time-independent) features.



The initial release also supports the following sampling methods for model estimation:

- **Markov-Chain Monte Carlo (MCMC)** as a full sampling method
- **Maximum a Posteriori (MAP)** as a point estimate method
- **Variational Inference (VI)** as a hybrid-sampling method on approximate distribution



## Quick Start


#### Installation 
`pip install orbit-ml`

Orbit requires PyStan as a system dependency. PyStan is licensed under GPLv3 , which is a free, copyleft license for software.

#### Data
`iclaims_example` is a dataset containing the weekly initial claims for US unemployment benefits against a few related google trend queries from Jan 2010 - June 2018. 


Number of claims are obtained from [Federal Reserve Bank of St. Louis](https://fred.stlouisfed.org/series/ICNSA) while google queries are obtained through [Google Trends API](https://trends.google.com/trends/?geo=US).

#### Quick Starter Code
```python
# import dependencies
import pandas as pd
import numpy as np
import warnings

from orbit.models.dlt import DLTMAP, DLTFull
from orbit.diagnostics.plot import plot_predicted_data

# Load data
df = pd.read_csv("../../examples/data/iclaims_example.csv", parse_dates=["week"])
df["claims"] = np.log(df["claims"])

# train test split
test_size = 104
train_df = df[:-test_size]
test_df = df[-test_size:]

# DLT model 
dlt = DLTMAP(
    response_col="claims",
    date_col="week",
    seasonality=52,
    seed=2020,
)

# fit
dlt.fit(df=train_df)

# predict
predicted_df = dlt.predict(df=test_df)

# plot
plot_predicted_data(training_actual_df=train_df, predicted_df=predicted_df,
                    date_col="week", actual_col="claims",
                    test_actual_df=test_df)

```
{{< figure src="/posts/ml/images/orbit-quickstart.png" >}}


## References

{{< admonition type=success title="Reference links" open=True >}}
- [Github](https://github.com/uber/orbit)
- [ArXiv Paper](https://arxiv.org/abs/2004.08492)
- [Documentation](https://uber.github.io/orbit/)
{{< /admonition >}}

