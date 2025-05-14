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

METHODOLOGY
A. Datasets and Preprocessing
This section describes the complete data preprocessing
pipeline, feature engineering strategy, and clustering methods
used for unsupervised ADL recognition using the Ordonez A
and B smart home datasets. Our approach focuses on modeling
the sensor activation stream as a sequence of related events,
from which we derive a set of temporal, spatial, and contextual
features to support unsupervised learned and clustering. The
experiments utilize two datasets acquired from Canvas,
OrdonezA_Sensors.txt and OrdonezB_Sensors.txt, each
corresponding to a different individual living independently in a
multi-room smart home environment. Both datasets record
ambient sensor activations over several weeks, covering a
variety of common activities of daily living. Each event in the
dataset includes start and end timestamps, sensor location, type,
and place (e.g., kitchen, bathroom). In addition to sensor data,
we also incorporate manually labeled ADLs provided in
OrdonezA_ADLs.txt and OrdonezB_ADLs.txt, which specify
the start and end times for each activity. These labels are not
used during clustering but serve to evaluate clustering
performance afterwards.
B. Temporal Relationships
To work with temporal dynamics, we first parsed the start
and end timestamps of each sensor activation into datetime
objects and computed each event's duration in seconds. Events
with missing timestamps were discarded. This step ensures that
each data point represents a valid time-bound sensor interaction.
The datasets from both users were merged into a unified event
stream. To retain the identity of each subject, a Set column was
added (denoting 'A' or 'B'). The merged data was sorted
chronologically by activation time. Additionally, we computed
the time since the last activation, which captures temporal gaps
between successive events, and filled any missing values with
zero for initialization.
C. Spatial/Transition Based Relationships
Each sensor's location (e.g., "Fridge", "Seat") was mapped
to a higher-level room category (e.g., "Kitchen", "Living
Room") using a predefined dictionary. The resulting Room
feature captures spatial context and was encoded numerically
using category encoding. This abstraction facilitates
understanding behavioral patterns at the room level rather than
individual sensor granularity. To capture temporal dependencies
and activity transitions, we derived several features based on
sensor sequences, such as the type of the previous sensor event
in the stream (PrevSensorType), the categorical combination of
previous and current sensor types (SensorTransition), and a
binary flag indicating whether the current sensor type differs
from the previous one (SensorTypeChange). Such transitionbased features are essential in modeling the flow of activities
and recognizing common routines.
D. Other Features
Two features were engineered to quantify behavioral
patterns over time, which are the average time interval between
consecutive activations of the same sensor location. This
reflects how frequently a space is used (SensorFrequency), and
the cumulative time gaps between events in the same room,
serving as a proxy for time spent in each space
(RoomOccupancy). These features aim to capture user routines
and dwell time in functional areas of the house.
To further enhance the feature space, one-hot encoding was
applied to the ‘Type’ of sensor, introducing a binary indicator
for each unique sensor type (e.g., PIR, Magnetic, Electric,
Pressure). This allowed the model to treat sensor types
independently rather than ordinally. In addition, a
SensorActivationCount feature tracks the number of times each
sensor location has been triggered up to a given point. This
helps identify areas of frequent usage and may correlate with
specific activities like cooking or grooming.
E. Clustering and Evaluation Metrics
While the clustering model operates in an unsupervised
manner, the ground-truth ADL labels were assigned post hoc
by matching each sensor activation to its corresponding activity
window from the labeled ADL files. This labeling was done per
timestamp and per subject, enabling evaluation of clustering
quality using standard metrics. All numerical features were
scaled using a StandardScaler, which centers each feature to
zero mean and unit variance. This step is critical to prevent
features with large numeric ranges (e.g., durations, time gaps)
from dominating the clustering algorithm. From this, the main
clustering algorithm evaluated here is DBSCAN (DensityBased Spatial Clustering of Applications with Noise): A
density-based method that discovers clusters of arbitrary shape
and can handle outliers (noise). Hyperparameters eps and
min_samples were tuned based on ARI/NMI performance. The
following external clustering evaluation metrics were used:
Adjusted Rand Index (ARI), which measures the similarity
between the clustering result and the ground truth, adjusted for
chance, and Normalized Mutual Information (NMI), which
quantifies the mutual dependence between cluster assignments
and true labels, normalized between 0 and 1. These metrics
allow assessment on how well the clustering aligns with the true
activity structure, even though labels were not used during
clustering.
