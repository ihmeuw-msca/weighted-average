What is WeAve?
================

The WeAve package smooths data across multiple dimensions using weighted
averages with methods inspired by the spatial-temporal models developed in the
following paper:

Foreman, K.J., Lozano, R., Lopez, A.D., et al. "`Modeling causes of death: an
integrated approach using CODEm <https://pophealthmetrics.biomedcentral.com/articles/10.1186/1478-7954-10-1>`_",
Popul Health Metrics, vol. 10, no. 1, pp. 1-23, 2012.

Weighted Averages with WeAve
----------------------------

The WeAve package generalizes the spatial-temporal models in CODEm to smooth
data across mutliple dimensions using weighted averages. Users can specify
dimensions using the :doc:`Dimension <../api_reference/weave.dimension>` class,
where :doc:`distance <../api_reference/weave.distance>` and
:doc:`kernel <../api_reference/weave.kernels>` functions determine how
weights are calculated.

Distance functions :math:`d(x_i, x_j)` calculate the distance between points
:math:`x_i` and :math:`x_j`, and kernel functions
:math:`k(d_{i, j} \, ; p_1, p_2, \dots)` calculate smoothing weights given
distance :math:`d_{i, j}` and a set of parameters :math:`p_1, p_2, \dots`. In
WeAve, you can choose from three distance functions and five kernel functions.

Weighted averages :math:`\hat{y}` of dependent variables :math:`y` are
calculated using the :doc:`Smoother <../api_reference/weave.smoother>` class
for observations :math:`i` in data set :math:`\mathcal{D}` with

.. math:: \hat{y}_i = \sum_{j \in \mathcal{D}} w_{i, j} \, y_j,

where weights are first calculated for dimensions :math:`m \in \mathcal{M}`,
then multiplied,

.. math:: \tilde{w}_{i, j} = \prod_{m \in \mathcal{M}} w_{i, j}^m,

and finally normalized,

.. math:: w_{i, j} = \frac{\tilde{w}_{i, j}}{\sum_{k \in \mathcal{D}}
          \tilde{w}_{i, k}}.

If the standard deviation of each observation is known, then the final weights
are defined as

.. math:: w_{i, j} = \frac{\tilde{w}_{i, j} / \sigma_j^2}
         {\sum_{k \in \mathcal{D}} \tilde{w}_{i, k} / \sigma_k^2},

and the standard deviation of the smoothed observations are defined as

.. math:: \hat{\sigma_i} = \sqrt{\sum_{j \in \mathcal{D}} w_{i, j}^2 \, \sigma_j^2}.

For instructions on how to get started, see the :doc:`Quickstart <quickstart>`.
For descriptions of the modules, objects, and functions included in WeAve, see
the :doc:`API Reference <../api_reference/index>`.

Depth Kernel Normalization
^^^^^^^^^^^^^^^^^^^^^^^^^^

Note that when the depth kernel is used, the preceding dimension weights are
normalized in groups based on the values of the depth kernel weights. This
corresponds to the CODEm framework where the product of age and time weights
are normalized in groups based on the location hierarchy before being
multiplied by the location weights. For example, if :math:`m_n` is a dimension
that uses the depth kernel, we let :math:`\mathcal{D}_{i, j}` be the set of all
indices :math:`k \in \mathcal{D}` such that
:math:`w_{i, k}^{m_n} = w_{i, j}^{m_n}`. Then the intermediate combined weights
are given by

.. math:: \tilde{w}_{i, j} = w_{i, j}^{m_n} \,
          \frac{\prod_{\ell < n} w_{i, j}^{m_\ell}}
          {\sum_{k \in \mathcal{D}_{i, j}} \prod_{\ell < n}
          w_{i, k}^{m_\ell}}.

Inverse-Distance Weights
^^^^^^^^^^^^^^^^^^^^^^^^

Inverse-distance weights are inspired by
`inverse-variance weighting <https://en.wikipedia.org/wiki/Inverse-variance_weighting>`_.
When the inverse kernel is used, scaled distances are combined over all
dimensions :math:`m \in \mathcal{M}` to create intermediate weights

.. math:: \tilde{w}_{i,j} = \frac{1}
    {\sum_{m \in \mathcal{M}} d_{i,j}^m / r^m + \sigma_i^2}.

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
second stage, the inspiration for WeAve, in more detail.

Spatial-Temporal Models
^^^^^^^^^^^^^^^^^^^^^^^

After the first stage linear mixed effects models have been run, the residuals
are calculated (observed - predicted dependent variable). It is assummed that
these residuals contain patterns that vary systematically across age, time, and
location that are not captured by the linear mixed effects models.
Spatial-temporal smoothing is applied to the residuals and the result added to
the first stage predictions in an effort to account for this additional
information.

For each observation :math:`i` in the data set :math:`\mathcal{D}`, weights
are assigned to the remaining observations :math:`j` based on their similarity
across age, time, and location. The predicted or smoothed residual
:math:`\hat{r}` is then the weighted average of the residuals :math:`r`,

.. math:: \hat{r}_i = \sum_{j \in \mathcal{D}} w_{i, j} \, r_j.

Weights are calculated based on similarity in age,

.. math:: w_{i, j}^a = \frac{1}{\exp(\omega \, |a_i - a_j|)}

similarity in year,

.. math:: w_{i, j}^t = \left(1 - \left(\frac{|t_i - t_j|}
          {\max_{k \in \mathcal{D}}|t_i - t_k| + 1}\right)^\lambda\right)^3,

and similarity in location,

.. math:: w_{i, j}^\ell = \begin{cases} \zeta & \text{same country}, \\
          \zeta(1 - \zeta) & \text{same region}, \\ (1 - \zeta)^2 &
          \text{same super region}, \\ 0 & \text{otherwise}, \end{cases}

and then combined into a single weight. Specifically, let
:math:`\mathcal{D}_{i, j}` be the set of all indices :math:`k \in \mathcal{D}`
such that :math:`w_{i, k}^\ell = w_{i, j}^\ell` (e.g., if points :math:`i` and
:math:`j` belong to the same country, then set :math:`\mathcal{D}_{i, j}` will
include all points in said country, etc.). Then the combined weights are given
by

.. math:: \tilde{w}_{i, j} = w_{i, j}^\ell \, \frac{w_{i, j}^a \,
          w_{i, j}^t}{\sum_{k \in \mathcal{D}_{i, j}} w_{i, k}^a \, w_{i, k}^t}.

Finally, weights are normalized so that all weights for each observation
:math:`i` sum to one,

.. math:: w_{i, j} = \frac{\tilde{w}_{i, j}}{\sum_{k \in \mathcal{D}}
          \tilde{w}_{i, k}}.
