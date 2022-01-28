## SUMMARY OF THE PAPER

Finding the best split points in the decision trees drives GBDT complexity. Splitting after value binning, as opposed to at every value, is more efficient. Because binning costs O(n_instances x n_feat), decreasing either would reduce binning complexity and, by extension, overall GBDT computation, key to applying them to large datasets.

A modification to feature binning for splitting that handles zero values in a sparse context is introduced.

In terms of instances, most boosting implementations are based on adaptive boosting, which applies weights to instances. Those based on gradient boosting perform stochastic instance sampling, which is not optimal. With the introduced GOSS, under-trained instances are always kept for subsequent tree training. The rest are sampled and applied a multiplier, compensating the effect of the sampling on the overall distribution. Estimating split variance gains from the GOSS-sampled instances greatly reduces computation while outperforming random sampling and usually outperforming straight random sampling.

In terms of features, the typically applied feature reduction techniques to reduce the space assume high feature redundancy, which not all datasets present. Instead, the introduced EFB reduces the space by bundling together mutually-exclusive features, common in large sparse feature spaces. Feature bundling can be reduced to the NP-hard problem of graph coloring, where vertices are features and edges exist among pairs of non-mutually-exclusive features. A good enough coloring scheme could then retain most training accuracy. Alternatively, edge weights can represent the level of conflict among feature pairs, allowing for a O(feat_2) greedy algorithm that iteratively bundles features starting from those with greater degrees. Just bundling features starting from those with less nonzero values results in a less complex greedy algorithm, as there is no need to build a graph. Once a pair of features is selected for bundling, values of one of them are shifted to avoid conflicts.

On both dense and sparse datasets, setting GOSS to always keep 5-10% of the highest-gradient instances and EFB not allowing conflicts, LightGBM overperforms XGBoost in time, memory and accuracy. GOSS is seen to improve speed-up less than linearly over sampling ratio and to be better that stochastic sampling. EFB is seen to greatly speed-up training for large sparse feature spaces that leverages the proposed feature binning.

## QUICK LEARNING LINKS

These links will give you some background on boosting to better read through the paper:

* GB basics: https://machinelearningmastery.com/gentle-introduction-gradient-boosting-algorithm-machine-learning/
* GB from scratch: https://medium.com/mlreview/gradient-boosting-from-scratch-1e317ae4587d
* AdaBoost: https://sefiks.com/2018/11/02/a-step-by-step-adaboost-example/
* AdaBoost vs GB: https://www.analyticsvidhya.com/blog/2020/10/adaboost-and-gradient-boost-comparitive-study-between-2-popular-ensemble-model-techniques/
* Stumps: https://github.com/parrt/msds689/blob/master/notes/stumps.ipynb
* Information gain: https://towardsdatascience.com/entropy-and-information-gain-in-decision-trees-c7db67a3a293

Links about more concrete aspects:

* Sampling vs weighting: https://stats.stackexchange.com/questions/168935/sampling-gradient-boosting-tree
* EFB and OHE: https://datascience.stackexchange.com/questions/41907/lightgbm-why-exclusive-feature-bundling-efb

## TYPES OF QUESTIONS

These are different angles on what can be asked. Some may not be as interesting as others to figure out the suitability of the candidate:

* The history
* The statistics
* The algorithmics
* The implementation
* The practicality
* The data science

## POTENTIAL CONCRETE QUESTIONS

### Generic questions about boosting

* What is a weak learner? How does it relate to bias vs variance? Key: low bias and high variance.
* Why do we use weak learners in boosting? Key: trained in sequence, have been proved to form a strong learner.
* What is a stump? How is the best split found using a stump? Key: single-split tree; find lowest loss across assessed splits.
* Why is learning rate also called shrinkage? Key: influence of a tree in the sequence shrinks over that of preceding tree.
* What is the difference with bagging? Key: parallel vs sequential ensembling.
* What is information gain? How does it relate to splitting? Key: reduction in entropy; a good split increases information gain, resulting in high variance gain.
* What is the gradient of an instance? Key: how under-trained it is.

### GOSS

* What challenge of large datatsets does the GOSS algorithm address? Key: high number of instances -> reduction of data instances
* What is the relationship between GOSS and purely stochastic gradient boosting (SGB)? Key: high information gain instances are always kept.
* What is the relationship between GOSS and AdaBoost's weights? Key: GOSS selects based on instance information gain and stochasticity; AdaBoost selects based on mathematically-derived weights.
* If you had to plot how the probability of keeping an instance depends on its information gain, how would the graph look like? Key: step function (a.k.a. flat piecewise function).
* What alternatives to GOSS can you think of? Key: some other sampling function than step function.
* How does GOSS avoid the change of the distribution of the data when removing instances with a small gradient? Key: random removal + multiply remaining data instances with small gradients with a constant value

### EFB

* What challenge of large datasets does the EFB algorithm address? Key: high dimensionality
* Is graph building used by the algorithm? Key: no, the proposed optimization does not, despite all the graph talk.
* How is the problem of maintaining the distribution of the data addressed (after merging mutually-exclusive features)? Key: allow exclusive features to reside in different bins

### Paper critical review

* Is the selection of datasets representative enough? Why?

### Experience with boosting algorithms?

* What is your experience with boosting algorithms? Any preference (e.g., LightGBM vs XGBoost)? Why?

