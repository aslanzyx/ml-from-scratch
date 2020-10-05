# Detection of Outlier

## Sanity Checking
- A type error
- Range of numerical features
- Unique entries for a categorical feature
- Duplicate features
  
## Model-based Outlier Detection
- Fit a probablistic model(norm dist and threshold the z-score)
- Outliers are examples with low brobablility

## Glabal vs Local Outliers 

## Graphical Outlier Detection
- Human decide outliers based on plots
- Box plots
- Scatter plots

## Distance-Based Outlier Detection
- Global
- For each point, compute the average distance to its KNN
- Threshold the outliers based on their average distances
- Local
  - Outlierness ratio: ${mean[d[:,x_i]] \over mean[d[neighbor(x_i),x_i]]}$

## Isolation Forest
- Grow a tree with random stump
- Stop when an example is isolated and mark it as an outlier
- Stop when an isolation score is met
- The isolation score is the depth of the tree (before one leaf has been isolated)

## Supervised Outlier Detection