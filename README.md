# Predicting and Clustering Flood Claim Severity and Storm Damage (2017)

Weekly Jupyter notebooks for Milestone Two of the Boston University Data Science Capstone (Module C, Fall 2026). Each week applies one group of machine learning methods to two 2017 natural disaster datasets, the FEMA National Flood Insurance Program (NFIP) redacted claims and the NOAA Storm Events database.

The project asks what drives the severity of flood-related losses. Milestone One showed that losses can be explained from the building, the policy, the flood, and the storm instead of the payout itself, and that both datasets are clustered by storm. Milestone Two returns to the two classification questions from Milestone One and then uses clustering to look for the natural groups in the data.

- Does an NFIP building payout exceed $50,000? (53% of 101,042 claims)
- Does a storm event report any property damage? (35% of 31,863 events)
- Without any labels, do claims and storm events group into the disasters that caused them?

## Notebooks

There are two notebooks per week, one for each dataset. `Week9-Storm-NFIP-Luu-Austin.ipynb` combines both Week 9 notebooks into one file.

| Week | Topic | Notebooks | Main result |
|---|---|---|---|
| 8 | K-nearest neighbors and distance metrics | `Week8-NFIP-Luu-Austin.ipynb`, `Week8-Storm-Luu-Austin.ipynb` | Tuned KNN reaches a test ROC-AUC of 0.861 (NFIP) and 0.888 (Storm Events). Scaling and the distance metric decide who counts as a neighbor, and random folds overstate Storm Events KNN by up to 0.07 |
| 9 | Gradient boosting, learning rate, number of estimators, tree depth, and regularization | `Week9-NFIP-Luu-Austin.ipynb`, `Week9-Storm-Luu-Austin.ipynb` | Tuned boosting reaches 0.880 and 0.920, the best results of the project so far |
| 10 | K-means, elbow method, silhouette score, feature scaling, and distance metrics | `Week10-NFIP-Luu-Austin.ipynb`, `Week10-Storm-Luu-Austin.ipynb` | Clusters of claim locations match FEMA's flood event names (adjusted Rand index 0.876), while storm events form only weak regional and seasonal regimes |
| 11 | DBSCAN, hierarchical agglomerative clustering, linkage methods, and dendrograms | `Week11-NFIP-Luu-Austin.ipynb`, `Week11-Storm-Luu-Austin.ipynb` | DBSCAN separates the 2017 disasters from scattered claims that rarely pay over $50,000, and space-time clustering shows that NOAA episodes stop at forecast office boundaries while storms do not |

## How the results were checked

- The classification notebooks (Weeks 8 and 9) use the same features and test sets as Milestone One. Storm Events models are always tested on whole storm episodes they never saw in training, since events from the same storm share weather and reporting.
- The clustering notebooks (Weeks 10 and 11) have no test set. The clusters are checked against outside labels the methods never saw, FEMA's flood event names and NOAA's storm episodes, and against split-half and resampling stability.

## Data

The raw data files are not included in this repository because of their size (the NFIP file is about 1 GB). To run the notebooks, download them and place them in the same folder as the notebooks.

- **FEMA NFIP redacted claims (v2)**, saved as `FimaNfipClaimsV2.csv`, from https://www.fema.gov/openfema-data-page/fima-nfip-redacted-claims-v2
- **NOAA Storm Events details for 2017**, saved as `StormEvents_details-ftp_v1.0_d2017_c20260519.csv`, from https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/ (unzip the 2017 details file. The `c` date in the file name may differ in newer downloads.)

The notebooks filter the NFIP data to losses in 2017.

## Running the notebooks

The notebooks were run with Python 3.14, pandas 2.3, NumPy 2.4, SciPy 1.17, matplotlib 3.10, and scikit-learn 1.8.

```bash
pip install pandas numpy scipy matplotlib scikit-learn jupyter
jupyter notebook
```

Every notebook is saved with its outputs, so the results and figures can be read on GitHub without running anything. Most notebooks run in a few minutes. The Week 8 NFIP grid search and the Week 9 notebooks take longer, about 10 to 20 minutes each.

## Related work

Milestone One (Weeks 1 to 6, regression, regularization, logistic regression, SVMs, and random forests) is in a separate repository: https://github.com/AustinLuu27/DX799---Milestone-One-

## Author

Austin Luu, Boston University, Fall 2026
