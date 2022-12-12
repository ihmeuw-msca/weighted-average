What is `weave`?
================

The `weave` package smooths data across multiple dimensions using
weighted averages with methods inspired by the spatial-temporal models
developed in the following paper:

Foreman, K.J., Lozano, R., Lopez, A.D., et al. "`Modeling causes of
death: an integrated approach using CODEm <https://pophealthmetrics.biomedcentral.com/articles/10.1186/1478-7954-10-1>`_",
Popul Health Metrics, vol. 10, no. 1, pp. 1-23, 2012.

Notes
* The spatio-temporal models then utilize additional regression analysis to take into account how the dependent variable further varies across time, space, and age.
* Local regression in three dimensions
* Assumes that residuals contain valuable information that cannot be directly observed but nonetheless vary systematically across geographic region, time, and age group.

Local regression is different than weighted average
* Local regression solves for each point using weighted least squares on nearby subset of the data point
* Usually first or second degree; using 0 means moving weighted average (!)

Should I connect it to CODEm (i.e., discuss residual smoothing?) or just general weighted averages?
Should I use a specific data set for the Quickstart? 