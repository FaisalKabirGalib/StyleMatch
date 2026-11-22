# StyleMatch

Comparative Fashion Product Recommendation Using Collaborative Filtering and Matrix Factorization.

**Interactive Demo (GitHub Pages):** https://faisalkabirgalib.github.io/StyleMatch/

This research starter ranks fashion products for existing users. It includes the downloaded Amazon 2014 Clothing, Shoes and Jewelry 5-core research dataset, a reproducible offline experiment, a printable progress report, and an interactive browser demo (`docs/`) powered by exported outputs from the trained models. Product images, outfit matching, and a deployed backend app remain future extensions.

## Run

```bash
cd /home/galib/Dev/StyleMatch
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python scripts/download_data.py
.venv/bin/python scripts/benchmark.py
.venv/bin/python scripts/build_report.py
.venv/bin/python scripts/verify_results.py
.venv/bin/python scripts/build_demo_data.py
```

The default pilot evaluates up to 1,000 validation users and 3,000 test users, sampled with seed 42. To evaluate every eligible user:

```bash
.venv/bin/python scripts/benchmark.py --test-users 0 --validation-users 0
.venv/bin/python scripts/build_report.py
```

Rerunning replaces generated splits, metrics, and reports. Keep separate output copies for experiments you want to compare.

## Files

- `data/raw/`: original compressed review file, unchanged.
- `data/source_manifest.json`: download source, SHA-256 checksum, row count, and license note.
- `data/processed/`: chronological CSV partitions, sparse training matrix, and user/product index mappings.
- `scripts/benchmark.py`: Random, Popularity, ItemKNN, truncated SVD, and ItemKNN–Popularity Blend.
- `results/metrics.csv`: aggregate test metrics; values are fractions, not percentages.
- `results/per_user_metrics.csv`: individual metric values for reproducibility and paired comparisons.
- `results/run.json`: cutoffs, counts, validation search, bootstrap interval, environment, and parameters.
- `results/recommendation_examples.json`: sample product recommendations and held-out targets.
- `results/comparison.png` and `.svg`: shareable result figures.
- `results/progress_report.html`: progress report; open in a browser and print to PDF.
- `results/summary.html`: compact comparison for the conversation.

## Evaluation

Deduplicate user–product pairs, retaining the first review. Split all interactions by global time at approximately 80% and 90%. Ratings of 4 or 5 become implicit positive preferences. Low ratings do not train the positive interaction matrix, but previously reviewed products of any rating are excluded from recommendations. All models use the same frozen training window.

Validation selects the blend weight from 0.25, 0.50, and 0.75 using NDCG@10. The test set is used once for the reported comparison. Evaluate users with at least three training reviews and at least one relevant test product present in the training-positive catalog. Rank the entire training-positive catalog after excluding earlier reviews. New users and new products are outside this experiment; their excluded-event counts appear in `run.json`.

Report macro-averaged Precision@10, Recall@10, NDCG@10, MRR@10, and catalog coverage. A paired user bootstrap estimates uncertainty for the prespecified Blend-versus-Popularity NDCG difference. This interval describes this fixed pilot and does not replace multiple temporal experiments. Do not compare these values directly with papers using different splits or sampled negatives.

The blend combines a behavioral model and popularity. It does not yet combine image/text content with collaborative filtering. SVD here is a 64-component truncated decomposition of the binary interaction matrix, rather than a supervised rating-prediction model.

## Research sources

- [Amazon 2014 source and attribution](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html). Public research download; no standalone permissive data license is specified on the source page.
- [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/): a newer option for later scaling.
- [H&M dataset and rules](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/rules): an alternative with competition-specific access and academic/noncommercial use conditions. Not downloaded here.
- [RecBole](https://github.com/RUCAIBox/RecBole): reusable recommendation framework for a wider algorithm comparison.
- [implicit](https://github.com/benfred/implicit): reusable collaborative filtering implementations, including ALS and BPR.
- [FashoReco](https://github.com/aldensiol/FashoReco): an existing H&M student project to study for project structure; inspect its licensing before reusing code.
- [Fashion RecSys Pipeline](https://github.com/TanyaIgnatenko/Fashion-RecSys-Pipeline): an example of retrieval followed by reranking.
- [A Review of Modern Fashion Recommender Systems](https://arxiv.org/abs/2202.02757): literature background and alternative fashion recommendation tasks.

The implementation in this folder is a new starter experiment; it does not reproduce or claim the published scores of these projects.
