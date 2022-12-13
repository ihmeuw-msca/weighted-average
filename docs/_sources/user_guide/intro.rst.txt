What is `weave`?
================

The `weave` package smooths data across multiple dimensions using weighted
averages with methods inspired by the spatial-temporal models developed in the
following paper:

Foreman, K.J., Lozano, R., Lopez, A.D., et al. "`Modeling causes of death: an
integrated approach using CODEm <https://pophealthmetrics.biomedcentral.com/articles/10.1186/1478-7954-10-1>`_",
Popul Health Metrics, vol. 10, no. 1, pp. 1-23, 2012.


Cause of Death Ensemble Model
-----------------------------

The Cause of Death Ensemble model (CODEm) predicts cause-specific mortality for
a large number of countries. As an ensemble model, it combines results from
many model families to produce predictions with improved accuracy. Two of the
model families included in CODEm, called spatial-temporal models, use weighted
averages to smooth results across age, time, and location.

These spatial-temporal models are created in three stages. First, either the
logit of the cause fraction or the natural log of the death rate is modeled
using a linear mixed effects model. Next, spatial-temporal smoothing is used to
account for additional variations across age, time, and location. Finally,
Gaussian process regression is applied to predict uncertainty. We describe the
second stage, the inspiration for `weave`, in more detail.

Spatial-Temporal Models
^^^^^^^^^^^^^^^^^^^^^^^

After the first stage linear mixed effects models have been run, the residuals
are calculated (predicted - observed dependent variable). It is assummed that
these residuals contain patterns that vary systematically across age, time, and
location that are not captured by the linear mixed effects models.
Spatial-temporal smoothing is applied to the residuals and the result added to
the first stage predictions in an effort to account for this important
information.

For each observation :math:`i` in the data set :math:`\mathcal{D}`, weights
are assigned to the remaining observations :math:`j` based on their similarity
across age, time, and location. The predicted or smoothed residual
:math:`\hat{r}` is then the weighted average of the residuals :math:`r`,

.. math:: \hat{r}_i = \sum_{j \in \mathcal{D}} w_{i, j} \, r_j.

Weights are calculated based on similarity in age,

.. math:: w_{a_{i, j}} = \frac{1}{\exp(\omega \, |a_i - a_j|)}

similarity in year,

.. math:: w_{t_{i, j}} = \left(1 - \left(\frac{|t_i - t_j|}
          {\max_{k \in \mathcal{D}}|t_i - t_k| + 1}\right)^\lambda\right)^3,

and similarity in location,

.. math:: w_{\ell_{i, j}} = \begin{cases} \zeta & \text{same country}, \\
          \zeta(1 - \zeta) & \text{same region}, \\ (1 - \zeta)^2 &
          \text{same super region}, \\ 0 & \text{otherwise}, \end{cases}

and then combined into a single weight, 

.. math:: \tilde{w}_{i, j} = w_{\ell_{i, j}} \, \frac{w_{a_{i, j}} \,
          w_{t_{i, j}}}{\sum_{k \in \mathcal{D}} w_{a_{i, k}} \,
          w_{t_{i, k}}}.

Finally, weights are normalized so that all weights for each observation
:math:`i` sum to one,

.. math:: w_{i, j} = \frac{\tilde{w}_{i, j}}{\sum_{k \in \mathcal{D}}
          \tilde{w}_{i, k}}.

Weighted Averages with `weave`
------------------------------

The `weave` package generalizes the spatial-temporal models in CODEm to smooth
data across mutliple dimensions using weighted averages. Users can specify
dimensions using the :doc:`Dimension <../api_reference/weave.dimension>` class,
where :doc:`distance <../api_reference/weave.distance>` and
:doc:`kernel <../api_reference/weave.kernels>` functions determine how
weights are calculated.

Distance functions :math:`d(x, y)` calculate the distance between points
:math:`x` and :math:`y`, and kernel functions :math:`k(d; r)` calculate
smoothing weights given distance :math:`d` and a set of parameters :math:`r`.
In `weave`, you can choose from three distance functions and four kernel
functions.

Weighted averages are calculated using the
:doc:`Smoother <../api_reference/weave.smoother>` class for observations
:math:`i` in data set :math:`\mathcal{D}` with

.. math:: \hat{x}_i = \sum_{j \in \mathcal{D}} w_{i, j} \, x_j,

where weights are first calculated for dimensions :math:`a, b, c, \dots`, then
multiplied,

.. math:: \tilde{w}_{i, j} = w_{a_{i, j}} \, w_{b_{i, j}} \, w_{c_{i, j}}
          \cdots,

and finally normalized,

.. math:: w_{i, j} = \frac{\tilde{w}_{i, j}}{\sum_{k \in \mathcal{D}}
          \tilde{w}_{i, k}}.

For instructions on how to get started, see the :doc:`Quickstart <quickstart>`.
For descriptions of the modules, objects, and functions included in `weave`,
see the :doc:`API Reference <../api_reference/index>`.
