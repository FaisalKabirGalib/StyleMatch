"""Build the printable report and figures directly from measured experiment files."""
import base64
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results'
run = json.loads((OUT / 'run.json').read_text())
manifest = json.loads((ROOT / 'data/source_manifest.json').read_text())
scores = pd.read_csv(OUT / 'metrics.csv').set_index('model')
best = scores.index[0]
pop = float(scores.loc['Popularity', 'ndcg_at_10'])
gain = (float(scores.iloc[0].ndcg_at_10) / pop - 1) * 100 if pop else float('nan')
low, high = run['blend_minus_popularity_ndcg_paired_bootstrap_95_ci']
confidence = ('The interval is above zero, supporting a positive difference for this fixed experiment.'
              if low > 0 else 'The interval includes zero; this experiment does not establish a positive difference.'
              if high >= 0 else 'The interval is below zero, supporting a negative difference for this fixed experiment.')

fig, ax = plt.subplots(figsize=(9, 4.6), layout='constrained')
values = scores.ndcg_at_10.to_numpy()
bars = ax.barh(scores.index, values, color=['#0f766e'] + ['#64748b'] * (len(scores) - 1))
ax.invert_yaxis()
ax.bar_label(bars, labels=[f'{v:.5f}' for v in values], padding=5)
ax.set_xlim(0, max(values) * 1.25)
ax.set_xlabel('NDCG@10 (higher is better)')
ax.set_title(f'StyleMatch: measured results on {run["test"]["evaluated_users"]:,} test users', loc='left', pad=18)
ax.spines[['top', 'right']].set_visible(False)
ax.xaxis.grid(True, alpha=.18)
ax.set_axisbelow(True)
fig.savefig(OUT / 'comparison.png', dpi=180)
fig.savefig(OUT / 'comparison.svg')
plt.close(fig)

columns = [('precision_at_10', 'Precision@10'), ('recall_at_10', 'Recall@10'),
           ('ndcg_at_10', 'NDCG@10'), ('mrr_at_10', 'MRR@10'), ('catalog_coverage', 'Coverage')]
table = '<div class="table-wrap"><table><caption>Measured test metrics; fractions between 0 and 1, higher is better.</caption><thead><tr><th scope="col">Model</th>'
table += ''.join(f'<th scope="col">{label}</th>' for _, label in columns) + '</tr></thead><tbody>'
for model, row in scores.iterrows():
    table += f'<tr><th scope="row">{html.escape(model)}</th>'
    table += ''.join(f'<td>{row[key]:.5f}</td>' for key, _ in columns) + '</tr>'
table += '</tbody></table></div>'

style = '''
html,body{background:transparent;color:var(--foreground,#17212b);font-family:var(--font-sans,system-ui,sans-serif);line-height:1.6;margin:0}
*{box-sizing:border-box}h1{font-size:28px;line-height:1.2;margin:0 0 12px}h2{font-size:20px;margin:28px 0 8px}h3{font-size:17px;margin:18px 0 6px}p{margin:8px 0 14px}
a{color:var(--accent-foreground,#0f766e);text-underline-offset:3px}small,.muted{color:var(--muted-foreground,#536271)}
.facts{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin:20px 0}.fact{background:var(--secondary,#eef4f4);padding:16px;border-radius:var(--radius,8px)}.fact strong{display:block;font-size:23px}.fact span{font-size:14px}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:13px}caption{text-align:left;margin-bottom:8px;font-size:14px}th,td{padding:10px 6px;border-bottom:1px solid var(--border,#d6dfe2);text-align:right}th:first-child{text-align:left}thead th{font-size:12px}tbody tr:first-child{font-weight:700}.table-wrap{overflow:auto}
img{max-width:100%;height:auto}ol,ul{padding-left:24px}li{margin:6px 0}code{font-size:13px;overflow-wrap:anywhere}details{margin:12px 0}summary{cursor:pointer;padding:10px 0;font-weight:650;min-height:44px}a:focus-visible,summary:focus-visible{outline:3px solid var(--ring,#0f766e);outline-offset:3px}
@media(max-width:460px){.facts{grid-template-columns:1fr}.fact strong{font-size:22px}h1{font-size:24px}table{font-size:12px}th,td{padding:8px 4px}}
@media print{body{color:#111}a{color:#111}.table-wrap{overflow:visible}h2,h3{break-after:avoid}table,img,.fact{break-inside:avoid}details>*{display:block}summary{display:none}}
'''

source = '<a href="https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html">official Amazon dataset page</a>'
intro = f'''<h1>StyleMatch</h1><p><strong>Comparative Fashion Product Recommendation Using Collaborative Filtering and Matrix Factorization</strong></p>
<p>Progress report · {run['run_utc'][:10]} · Initial offline experiment</p>
<p>The project recommends fashion products from previous user preferences and compares algorithms under the same evaluation conditions. <strong>{best}</strong> obtained the highest test NDCG@10 in this pilot. These measurements support a reproducible progress submission; wider validation and a user-facing prototype remain future work.</p>'''
facts = f'''<div class="facts"><div class="fact"><strong>{run['raw_reviews']:,}</strong><span>source reviews</span></div><div class="fact"><strong>{run['candidate_items']:,}</strong><span>candidate products</span></div><div class="fact"><strong>{run['test']['evaluated_users']:,}</strong><span>evaluated test users</span></div></div>'''
protocol_short = f'''<p>Global chronological split; ratings ≥4 are positive preferences. Models rank the full training-positive catalog, excluding products already reviewed. The blend weight was selected on {run['validation']['evaluated_users']:,} validation users. All models use the same frozen training window.</p>'''
sources = '''<ul>
<li><a href="https://github.com/aldensiol/FashoReco">FashoReco</a>: an existing H&amp;M student project for studying project structure and recommendation workflows; check its license before copying code.</li>
<li><a href="https://github.com/RUCAIBox/RecBole">RecBole</a>: a reusable framework for expanding the algorithm comparison.</li>
<li><a href="https://github.com/benfred/implicit">implicit</a>: collaborative filtering implementations suitable for adding ALS and BPR.</li>
<li><a href="https://arxiv.org/abs/2202.02757">A Review of Modern Fashion Recommender Systems</a>: background for the literature review and the distinction between product, outfit, and size recommendation.</li>
</ul>'''
next_steps = '''<ol>
<li><strong>Expand the comparison:</strong> add implicit ALS or BPR and test several factor counts and neighborhood sizes using validation data.</li>
<li><strong>Add fashion content:</strong> obtain the matching product metadata, filter to this catalog, and compare title/category TF-IDF against collaborative models. Then test a content–collaborative hybrid. Do not use held-out review text as product features.</li>
<li><strong>Strengthen the evidence:</strong> run all eligible users, repeat across global time cutoffs, report uncertainty, and evaluate new users/items separately. Reserve a new untouched test window for later model development.</li>
<li><strong>Build the demonstration:</strong> add a small interface with product names and images after obtaining suitable metadata. Explain recommendations and show the selected user's history.</li>
<li><strong>Finalize the submission:</strong> add student/institution details, objectives, related work, methodology, the measured comparison, limitations, and the next milestone. Export this report to PDF through browser printing.</li>
</ol>'''

body = intro + facts + '<h2>Problem and objectives</h2><p>Online fashion catalogs contain many products. The research question is whether personalized behavioral models recover later highly rated products more effectively than recommending the same popular products to everyone. The first objectives are dataset preparation, reproducible baselines, an algorithm comparison, and measured ranking quality.</p>'
body += f'''<h2>Dataset and preparation</h2><p>Downloaded the Amazon 2014 Clothing, Shoes and Jewelry 5-core file from the {source}. The local file contains {run['raw_reviews']:,} reviews, {run['raw_unique_users']:,} users and {run['raw_unique_items']:,} products. User ID, product ID, rating and timestamp are used in this experiment.</p>
<p>The source covers {run['date_range_utc'][0][:10]} to {run['date_range_utc'][1][:10]} in the downloaded category. It is historical research data. The source page does not specify a standalone permissive data license, so it should be described as publicly available research data.</p>
<p>Retain each user–product pair's earliest review, producing {run['unique_user_item_pairs']:,} unique pairs. Training contains {run['split_rows']['train']:,} reviews, validation {run['split_rows']['validation']:,}, and test {run['split_rows']['test']:,}. The time boundaries are {run['cutoffs_utc'][0][:10]} and {run['cutoffs_utc'][1][:10]}. Positive training interactions total {run['positive_training_events']:,}.</p>'''
body += '''<h2>Implemented algorithms</h2><ul>
<li><strong>Random:</strong> reproducible random ranking to provide a basic reference.</li>
<li><strong>Popularity:</strong> rank products by the number of positive training reviews.</li>
<li><strong>ItemKNN:</strong> cosine similarity between product interaction columns, retaining 100 neighbors per product. Score products from the user's positive history. A tiny popularity term breaks equal scores.</li>
<li><strong>SVD:</strong> 64-component truncated singular value decomposition of the binary interaction matrix, with seven iterations and seed 42.</li>
<li><strong>Blend:</strong> normalize ItemKNN and popularity scores and combine them. This is a behavioral–popularity blend; product content is planned for a later hybrid.</li></ul>'''
body += '<h2>Experimental method</h2>' + protocol_short
body += f'''<p>Evaluate users with at least three training reviews and at least one positive held-out product in the training-positive catalog. The pilot samples at most 1,000 validation users and 3,000 test users with seed 42; {run['test']['eligible_users']:,} test users satisfy the eligibility rules. All algorithms are evaluated on the same selected users and targets.</p>
<p>Of {run['test']['positive_events']:,} positive test events, {run['test']['warm_user_and_item_positive_events']:,} have a user and item present in training. Further history and sampling rules determine the evaluated users. This warm-start comparison does not assess new-user or new-item performance.</p>
<p>Metrics are averaged per user. Precision measures the fraction of ten recommendations that are relevant; recall measures the fraction of relevant held-out products recovered; NDCG rewards relevant products near the top; MRR measures the position of the first relevant product; coverage is the fraction of candidate products recommended to at least one evaluated user.</p>
<p>The validation search tried ItemKNN weights 0.25, 0.50, and 0.75; the selected weight was <strong>{run['selected_knn_weight']:.2f}</strong>. Model and popularity features are fitted only on training interactions. For test recommendations, validation-period reviews are also excluded as previously seen products, but do not update model features.</p>'''
image = base64.b64encode((OUT / 'comparison.png').read_bytes()).decode()
body += '<h2>Measured results</h2>' + table + f'<img src="data:image/png;base64,{image}" alt="Horizontal bar chart of test NDCG at ten. The exact values are in the preceding table.">'
body += f'''<p><strong>{best}</strong> achieved NDCG@10 of <strong>{scores.iloc[0].ndcg_at_10:.5f}</strong>, compared with {pop:.5f} for popularity ({gain:+.1f}% relative difference). This is a ranking metric, not a classification accuracy percentage.</p>
<p>For the prespecified Blend-versus-Popularity comparison, mean NDCG difference is {run['blend_minus_popularity_ndcg_mean']:.5f}. A paired bootstrap over test users with 2,000 resamples gives a 95% interval of [{low:.5f}, {high:.5f}]. {confidence} It does not establish that the blend is best on other datasets or time periods.</p>'''
body += '''<h2>Progress and limitations</h2><p><strong>Completed:</strong> source research, dataset download and checksum, preprocessing, chronological splits, five ranking approaches, validation-only blend selection, test evaluation, metric export, sample recommendations, and reproducible charts.</p>
<p><strong>Pending:</strong> product metadata and visual recommendations, a content-based model, broader hyperparameter experiments, cold-start evaluation, repeated time-window validation, and an interactive demonstration. The current results use product IDs because product names and images have not been downloaded.</p>
<p>Reviews are proxies for preferences rather than purchase logs, and products without reviews are not verified negative examples. The historical 5-core subset favors active users and reviewed items. Low ranking scores on a large catalog are possible and should be reported honestly. Results from other repositories cannot be compared directly unless data, candidate sets, and split protocols match.</p>'''
body += '<h2>Related work and available implementations</h2>' + sources
body += '<p>For a later large-scale experiment, consider <a href="https://amazon-reviews-2023.github.io/">Amazon Reviews 2023</a>. The <a href="https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/rules">H&amp;M competition</a> is another relevant source, subject to its access rules and noncommercial/academic use conditions; it was not downloaded for this pilot.</p>'
body += '<h2>Next milestones</h2>' + next_steps
body += f'''<h2>Reproducibility and verification</h2><p>Results were generated locally, rather than copied from another project. Run details and package versions are in <code>results/run.json</code>; commands are in <code>README.md</code>. The benchmark asserts strict time ordering, disjoint relevant and seen products, ten unique unseen recommendations, finite selected scores, and metric bounds. Ranking helpers were checked against hand-calculated metric examples.</p><p>Source SHA-256: <code>{manifest['sha256']}</code>. Original compressed size: {manifest['bytes']:,} bytes. Raw user identifiers and review text remain in the local source archive; exported example recommendations use numeric user indexes.</p>'''

document = lambda content, title: '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>' + title + '</title><style>' + style + '</style></head><body>' + content + '</body></html>'
(OUT / 'progress_report.html').write_text(document('<main style="max-width:960px;margin:auto;padding:24px">' + body + '</main>', 'StyleMatch progress report'))
summary = f'''<p><strong>StyleMatch · Initial measured comparison</strong></p>''' + facts + table
summary += f'''<p><strong>{best}</strong> leads this pilot on NDCG@10. All numbers were measured locally using the same chronological split and full candidate catalog.</p>'''
summary += '<details><summary>How the results were measured</summary>' + protocol_short + '<p>Warm users and items only; sampled test users, seed 42. These are offline ranking results, not proof of sales improvement or overall superiority.</p></details>'
summary += '<details><summary>Existing projects and next steps</summary>' + sources + next_steps + '</details>'
(OUT / 'summary.html').write_text(document(summary, 'StyleMatch initial results'))
print(f'Report and figures generated from {OUT / "metrics.csv"}')
