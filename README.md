# Clustering-Activities-of-Daily-Living-without-Supervision-using-DBSCAN

This study explores the application of unsupervised
clustering techniques to smart home sensor data for identifying
some sample Activities of Daily Living (ADLs) without labeled
supervision. Using the publicly available CASAS Ordonez A and
B datasets, which record timestamped activations of various
ambient sensors in a two-person household, a set of contextual and
time-based features was engineered at the event level. These
features include sensor type, location, activation frequency, timeof-day encoding, inter-event timing, and room-level mappings, all
aimed at capturing behavioral patterns indicative of daily
routines. The overall clustering performance was evaluated using
external metrics—Adjusted Rand Index (ARI) and Normalized
Mutual Information (NMI)—against ground-truth activity labels.
Among the clustering algorithms tested, DBSCAN achieved the
most promising results (ARI = 0.320, NMI = 0.399), outperforming
K-Means and other baseline approaches, likely due to its
robustness to noise and ability to model complex clusters that
don’t solely rely on Euclidean distance. These preliminary
findings suggest that unsupervised approaches, when combined
with thoughtful feature engineering, can meaningfully uncover
latent structure in smart home sensor data, offering potential for
real-world activity recognition without manual annotation
