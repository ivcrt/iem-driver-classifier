# IEM Driver Type Classification

Predicting the driver technology of in-ear monitors from their
frequency-response curve using a small neural network. (Will try using a CNN later)

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
   This produces `data/iem_driver_dataset.csv` which has 909 labeled Frequency Response curves for 270 unique IEM models.

## Setup

```
pip install -r requirements.txt

```

## Usage

```
python build_iem_dataset.py   # build the labeled dataset
python [file name containing the word 'train'].py # train the different models and get the loss and accuracy graphs
```

## Nerdy Audiophile Glossary

| Term | Definition |
|------|------------|
| **IEM** | In-Ear Monitor — a small earphone that sits inside the ear canal, used for personal listening or on-stage monitoring. |
| **DD** | Dynamic Driver — the most common type; uses a voice coil and diaphragm (like a miniature speaker). Known for natural bass and cohesive sound. |
| **BA** | Balanced Armature — a compact driver borrowed from hearing aids; uses a tiny vibrating armature instead of a diaphragm. More precise in the mids and highs, but typically weaker in bass extension. |
| **Hybrid** | An IEM that combines two or more driver types (usually DD + BA) to leverage the strengths of each. |



