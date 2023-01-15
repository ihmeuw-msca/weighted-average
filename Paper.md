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
    orcid: 
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

Interpolation and prediction are a big challenge for big-data applications driven by sparse high dimensional datasets. 
For example, datasets used for the Global Burden of Disease study vary by year, age, sex, and location. Interpolation 
is required to estimate missing data, such as mortality or disease prevalence, when it is not available 
for particular year/age/sex/location groups. These datasets are also sparse, with non-random missing values: 
decades of data and/or entire age groups may be missing in multiple countries.

While there are many well-developed methods and packages for interpolation, including splines [https://pypi.org/project/ndsplines/] [@lyons2019ndsplines],  Loess [https://pypi.org/project/loess/] [@cappellari2013atlas3d], and 
general interpolation [https://pypi.org/project/pyinterp/], it is difficult to apply these methods to sparse high-dimensional datasets. 
First, support for Loess is limited to two dimensions, while support for pyinterp is limited to four. More importantly, when working with sparse datasets, sophisticated methods may not have enough information to find the `interpolated' solution, 
or may require a large number of decisions made by users, making them impractical. 

In practice, weighted averages of the data are a key approach for sparse high-dimensional datasets. Weighted average can be viewed 
as a simple case of Loess (taking a zeroth order local regression approach). Nevertheless, there are no packages that implement 
high-dimensional weighted averaging with customizable strategies for weighting. 

We fill this gap by implementing N-dimensinoal Weig hted Averaging (WeAve). WeAve provides a flexible interface to specify an 
arbitrary number of dimensions, distance functions to measure distances across these dimensions, and kernel functions used to 
smooth data across the dimensions. The package implements and generalizes methods that support widely published 
Global Burden of Disease studies, [cite GBD 2019], and detailed in [@foreman2012modeling]. 


# Statement of Need

Weighted averaging is a simple technique that allows working with challenging high-dimensional sparse datasets. 
Practitioners who work with such data have deep understanding of each dimension, and need a simple and clear way
to incorporate this understanding into the weighted averaging process. 

The `WeAve' package is designed to support any number of dimensions, allow practitioners to design their own distance functions, 
and to specify parametrized kernels that are used to smooth the data. Default settings implement distance functions
and kernels required to reproduce space-time smoothing, the backbone of Cause of Death Modeling [@foreman2012modeling] 
used for the Global Burden of Disease. However, the specification of all key elements is flexible, making the package
easy to adapt to any use case.  


# Core idea and structure of `WeAve`

Distance functions $d(x_i, x_j)$ calculate the distance between points $x_i$ and $x_j$, and kernel functions $k(d_{i, j} \, ; p_1, p_2, \dots)$ calculate smoothing weights given distance $d_{i, j}$ and a set of parameters $p_1$, $p_2, \dots$. In WeAve, you can choose from three distance functions and four kernel functions.

Weighted averages $\hat{y}$ of dependent variables y are calculated using the :doc:`Smoother <../api_reference/weave.smoother>` class for observations $i$ in data set $\mathcal{D}$ with

$\hat{y}_i = \sum_{j \in \mathcal{D}} w_{i, j} \, y_j,$
where weights are first calculated for dimensions $m \in \mathcal{M}$, then multiplied,

$\tilde{w}_{i, j} = \prod_{m \in \mathcal{M}} w_{i, j}^m$,
and finally normalized,

$w_{i, j} = \frac{\tilde{w}_{i, j}}{\sum_{k \in \mathcal{D}}
\tilde{w}_{i, k}}$.

## Reproducing Space-Time Smoothing in CODEm

A detailed example shows how to use `WeAve' to reproduce core functinality of 
space-time smoothing in Cause of Death modeling for the Global Burden of Disease.  

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

Weights are then combined and normalized to sum to $1$. 



# Ongoing Research and Dissemination

The manuscript [@foreman2012modeling] introduces the overarching CODEm framework, and 
specifies the weighted average strategy. Over 396 publications, most in high-impact journals such as the Lancet, have used this approach
to obtain interpolants for sparse high dimensional data sets. 
The most recent publication, [@vos2020global], analyzes the burden of disease from 1990-2019 in 204 countries and territories. 


# References
