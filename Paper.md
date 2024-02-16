---
title: 'weave: A Python Package for Weighted Averaging Across Multiple Dimensions'

tags:
  - Python
  - Kernel weighting
  - Loess
  - Multiple dimensions
  - Space-time smoothing 
  - Global health 

authors:
  - name: Kelsey Maass
    orcid: 0000-0002-9534-8901
    affiliation: 1
  - name: Peng Zheng
    orcid: 0000-0002-1875-1801
    affiliation: 2
  - name: Aleksandr Aravkin
    orcid: 0000-0003-3313-215X
    affiliation: "1, 2"

affiliations:
 - name: Department of Health Metrics Sciences, University of Washington
   index: 1
 - name: Department of Applied Mathematics, University of Washington
   index: 2
date: 01.12.2023
bibliography: paper.bib

---

# Summary

Interpolation and prediction are a significant challenge for big-data applications driven by sparse high dimensional datasets. 
For example, quanitities (e.g. mortality rates and disease prevalence) used within the Global Burden of Disease study vary by year, age, sex, and location. These datasets are sparse with non-random missing values; in extreme cases, decades of data or information for entire age groups may be missing in multiple countries based on standards and practices in data collection. As a result, interpolation is required to estimate missing data for many year/age/sex/location combinations. 

While there are many methods and packages for interpolation, including splines [https://pypi.org/project/ndsplines/] [@lyons2019ndsplines],  Loess [https://pypi.org/project/loess/] [@cappellari2013atlas3d], and 
geo-reference data [https://pypi.org/project/pyinterp/], it is difficult to apply these methods to sparse high-dimensional datasets. 
Support for Loess is limited to two dimensional data, while support for pyinterp is limited to four-dimensional data.  
When working with sparse datasets and high dimensions, sophisticated methods do not have enough information to find the interpolated solution, 
and the solutions may be unstable, particularly for datapoints close to the boundary.  

Simpler interpolation techniques, particularly weighted averages of the data, are a key  approach for sparse high-dimensional datasets. Weighted average can be viewed as a simple case of Loess (taking a zeroth order local regression approach). Nevertheless, there are no packages that implement 
high-dimensional weighted averaging with customizable weighting strategies. 

We fill this gap by implementing N-dimensinoal Weighted Averaging (WeAve). WeAve provides a flexible interface to specify an 
arbitrary number of dimensions, distance functions to measure distances across  dimensions, and kernel functions used to 
smooth data across the dimensions. The package implements and generalizes methods that support widely published 
Global Burden of Disease studies, [cite GBD 2019], and detailed in [@foreman2012modeling]. 


# Statement of Need

Weighted averaging is a simple technique that allows interpolation for high-dimensional sparse datasets. 
Practitioners who work with such data may have scientific insight into how related the data is across different dimensions, 
and need a simple and clear way to incorporate this understanding into the weighted averaging process. 

The `WeAve' package is designed to support any number of dimensions, allow practitioners to design their own distance functions, 
and to specify parametrized kernels that are used to smooth the data. Default settings implement distance functions
and kernels required to reproduce space-time smoothing, the backbone of Cause of Death Modeling [@foreman2012modeling] 
used for the Global Burden of Disease. However, the specification of all key elements is flexible, making the package
easy to adapt to any use case.  


# Core idea and structure of `WeAve'


WeAve provides an interface to specify how the weighted average is computed using distance functions, kernel functions, and their parameters. 
Distance functions $d(x_i, x_j)$ calculate the distance between points $x_i$ and $x_j$, and kernel functions $k(d_{i, j} \, ; p_1, p_2, \dots)$ calculate smoothing weights given distance $d_{i, j}$ and a set of parameters $p_1$, $p_2, \dots$. WeAve provides currently provides three distance functions and four kernel functions, and the user can specify additional functions as necessary. 

Weighted averages $\hat{y}$ of dependent variables y are calculated using the :doc:`Smoother <../api_reference/weave.smoother>` class for observations $i$ in data set $\mathcal{D}$ with

$\hat{y}_i = \sum_{j \in \mathcal{D}} w_{i, j} \, y_j,$
where weights are first calculated for dimensions $m \in \mathcal{M}$, then multiplied,

$\tilde{w}_{i, j} = \prod_{m \in \mathcal{M}} w_{i, j}^m$,
and finally normalized,

$w_{i, j} = \frac{\tilde{w}_{i, j}}{\sum_{k \in \mathcal{D}}
\tilde{w}_{i, k}}$.

## Reproducing Space-Time Smoothing in CODEm

A detailed example shows how to use `WeAve' to reproduce core functinality of space-time smoothing in Cause of Death modeling (CODEm) for the Global Burden of Disease. 

For each observation $i$ in the data set $\mathcal{D}$, weights are assigned to the remaining observations $j$ based on their similarity across age, time, and location. The predicted or smoothed residual $\hat{r}$ is then the weighted average of the residuals $r$,
$\hat{r}_i = \sum_{j \in \mathcal{D}} w_{i, j} \, r_j$.

CODEm uses an exponential kernel for age: 
$w_{i, j}^a = \frac{1}{\exp(\omega \, |a_i - a_j|)}$ tricubic kernel for year:

$w_{i, j}^t = \left(1 - \left(\frac{|t_i - t_j|}
{\max_{k \in \mathcal{D}}|t_i - t_k| + 1}\right)^\lambda\right)^3$,

and and a custom kernel for the discrete location dimension: 

$w_{i, j}^\ell = \begin{cases} \zeta & \text{same country}, \\
\zeta(1 - \zeta) & \text{same region}, \\ (1 - \zeta)^2 &
\text{same super region}, \\ 0 & \text{otherwise}, \end{cases}$

Weights are then combined and normalized to sum to $1$ as described above. 



# Ongoing Research and Dissemination

The manuscript [@foreman2012modeling] introduces the overarching CODEm framework, and 
specifies the weighted average strategy. Over 396 publications, most in high-impact journals such as the Lancet, have used this approach
to obtain interpolants for sparse high dimensional data sets. 
The most recent publication, [@vos2020global], analyzes the burden of disease from 1990-2019 in 204 countries and territories. 


# References
