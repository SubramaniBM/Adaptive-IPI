# Appendix — Enron Deduplication Methodology

## Method
To eliminate potential train-test overlap arising from duplicated corporate announcements in the Enron corpus, we applied TF-IDF cosine similarity deduplication.

## Process
1. The Enron training subset (1,000 emails) and the Enron held-out test set (1,000 emails) were separately loaded.
2. A joint TF-IDF matrix was constructed using scikit-learn's TfidfVectorizer (max_features=10,000, English stop words removed).
3. Cosine similarity was computed between all test emails and all training emails.
4. All test emails exhibiting maximum similarity > 0.85 to any training email were removed.

## Results
- Original held-out test set: 1,000 emails
- Removed (near-duplicates): 75 emails
- Final disjoint evaluation set: 925 emails
- Maximum cosine similarity in final set: 0.8501

## Justification
Near-duplicates arise naturally from the Enron corpus structure (company-wide announcements appear in multiple employees' inboxes). Removing them ensures the FPR measurement is computed exclusively on emails the model has never encountered in any form.
