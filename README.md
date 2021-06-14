This repository is part my master's thesis.

### Preliminary Abstract
------------------------
At AABI 2021 we introduced KL flow integration.
This thesis investigates how the algorithm performs on more challenging problems, such as Bayesian logistic regression and multi modal sampling problems.
In doing a serious of issues are identified and problem agnostic solutions are proposed and themselves investigated.
In particular we show how the time derivative of the KL divergence can be computed under various flows, allowing us to find ones better suited to integration.
We investigate the effects of posterior approximation, alternative gradient descent steps and of annealing of the target distribution.
On problems of Bayesian inference we also compare flow integration to thermodynamic integration and find that in some cases it compares quite favourably, especially owing to its insensitivity to the number of particles.
