# IEM Driver Type Classification

Predicting the driver technology of in-ear monitors from their
frequency-response curve using a small neural network. 

The interesting part of this project isn't its accuracy number, it's the question *how much does the frequency response curve tell you about the
hardware inside?* 
The answer for this question turns out to be quite a lot for telling the Balanced Armature driver technology apart from the Dynamic Driver one. It ends up being a lot less if we add the Hybrid Category, which is a mix of the two.

## Context and Problems

Each IEM is described by a 120-point frequency-response curve on the audible spectrum (20Hz -> 20KHz). The label is the driver technology.

I started with a **3-class** problem (DD / BA / Hybrid) because they were the most represented technologies in the dataset and the other ones didn't have enough data.
But the hybrid class ended up being too ambiguous: a hybrid combines a dynamic driver as well as balanced armatures, so its Frequency Response is in between the two pure types and it makes it very hard to tell it apart from the two other technologies.

I made a 3-class MLP model using TensorFlow that obtained a **0.55 macro-F1**, that had error mostly concentrated on the Hybrid class.

Because of that, I chose to study the **binary** problem: distinguishing the
two *pure* driver types, **DD vs BA**, that have more distinct signatures. 

### Building the dataset (adding labels)
The two tables are joined on the IEM name: an exact match first,
then **fuzzy matching** (`rapidfuzz`, `token_set_ratio ≥ 92`) with a threshold that
requires the model numbers to be consistent (so for example "Blessing 2" will never match Blessing 3"). 
Fuzzy matching led to almost doubling the usable data:

| | Exact match only | + Fuzzy match |
|---|---|---|
| Unique models | 270 | **399** |
| Labeled curves | 909 | **1 748** |

After filtering to the studied classes, we end up with **1 031 curves** for DD vs BA
(562 DD / 469 BA).

## General Method

- **Features:** the raw 120-point FR curve.
- **Grouped split.** The same IEM was measured by multiple headphone reviewers. So we need to make those measures dependent, otherwise we'll end up with the same IEM in both training in validation. I used `GroupShuffleSplit` by regrouping those measurements by model name.
- **Data augmentation.** Each training curve has two clones with a ±1-bin shift and Gaussian noise of +0.1dB. 
- **Class imbalance.** The weights are balanced according to the frequency of a class in the dataset.
- **Hyperparameter search with optuna** Hyperparameter search with Optuna (30 trials) was explored but did not improve multi-seed evaluation, most likely due to overfitting to the validation set on a small dataset.

## Models used for the comparison
 Logistic Regression, Random Forest, XGBoost, and a **1D CNN**.
## Results

All numbers on the same grouped split (validation = 203 IEMs for the binary
task, 353 for the 3-class task).

| Task / Model | Accuracy | Macro-F1 |
|---|---|---|
| 3-class (DD/BA/Hybrid) — MLP | 0.573 ± 0.018 | 0.550 ± 0.012 |
| Binary (DD vs BA) — Majority baseline | 0.685 | 0.406 |
| Binary — Logistic Regression | 0.724 | 0.683 |
| Binary — Random Forest | 0.724 | 0.672 |
| Binary — XGBoost | 0.695 | 0.633 |
| **Binary — 1D CNN (5 seeds)** | **0.802 ± 0.038** | **0.763 ± 0.051** |


- The CNN beats the baselines, so its additionnal complexity is needed to classify Frequency Response curves

## Biggest mistake I made

1. **Augmenting the data before the split** I didn't have knowledge about data augmentation so I just applied the method, tripled my dataset and obtained great accuracy on my *3 class MLP*. After re-reading my code I realised my mistake, I fixed it and the accuracy dropped *20%*. What happened is the validation set contained augmented and noisy copies.



## Data

Two public sources used locally **They have to be downloaded separately**.

- **Frequency-response curves** from [PEQHUB/Squig-Rank](https://github.com/PEQHUB/Squig-Rank) (`public/data/curves.json`),

- **Driver-type labels** — from Crinacle's (Prominent Youtuber in the Audiophile Community) public IEM ranking list
  ([BorisLestsov/Crinacle_IEM_vis](https://github.com/BorisLestsov/Crinacle_IEM_vis),
  `raw.txt`; original source: https://crinacle.com/rankings/iems/).

> We only use the ranking list data to obtain IEM + Driver Technology couples to then 
> add a "Driver Technology" Column to our dataset


### Regenerating the dataset

1. Clone [Squig-Rank](https://github.com/PEQHUB/Squig-Rank) and copy and paste
   `public/data/curves.json` into `data/`.
2. Download `raw.txt` from
   [Crinacle_IEM_vis](https://github.com/BorisLestsov/Crinacle_IEM_vis) into `data/`.
3. Run the build script:
   ```
   python build_iem_dataset.py
   ```
   This produces `data/iem_driver_dataset.csv` which has 1748 labeled Frequency Response curves for 399 unique IEM models.

## Setup

```

pip install -r requirements.txt

```

## Usage

```
python build_iem_dataset.py   # build the labeled dataset
python [file name containing the word 'train'].py # train the different models and get the loss and accuracy graphs
python predict.py # make a prediction using one 120-point RF curve (requires adding the file name in the script)
```

## Nerdy Audiophile Glossary

| Term | Definition |
|------|------------|
| **IEM** | In-Ear Monitor — a small earphone that sits inside the ear canal, for personal listening or on-stage monitoring. |
| **FR** | Frequency Response — how loud the IEM plays at each frequency; the curve used as input here. |
| **DD** | Dynamic Driver — the most common type; a voice coil + diaphragm (a miniature speaker). Natural bass, cohesive sound. |
| **BA** | Balanced Armature — a compact driver borrowed from hearing aids; a tiny vibrating armature instead of a diaphragm. Precise mids/highs, typically weaker bass extension. |
| **Hybrid** | An IEM combining two or more driver types (usually DD + BA) to leverage the strengths of each. |


