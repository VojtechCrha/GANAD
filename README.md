
# GANAD: Improving medical data synthesis with DP-GAN and Deep Anomaly Detection

This repository contains the implementation of the GANAD [framework](http://resolver.tudelft.nl/uuid:9ff9775e-54ce-4857-9463-618745544215).

## Abstract

> Ensuring the privacy of medical data in a meaningful manner is a complex task. This domain presents a plethora of unique challenges: high stakes, vast differences between possible use cases, long-established methods that limit the number of feasible solutions, and more. Consequently, an effective approach to ensuring the privacy of medical data must be easy to adopt, offer robust privacy guarantees, and minimize the reduction in data utility.

> The unique nature of medical data presents distinct challenges and also opportunities. We consider various types of correlations that significantly impact privacy guarantees. However, these correlations can also be used to train a model for removing anomalies and subsequently enhancing the utility of synthetic medical data.

> This thesis proposes a framework compatible with state-of-the-art approaches for differentially private dataset release based on the usage of Generative Adversarial Networks (GANs). Our framework uses a part of the privacy budget to train an unsupervised learning model to detect and remove anomalies. We evaluate the performance of the framework using a variety of machine-learning models and metrics. The final results show an improvement of up 13% compared to approaches not using our framework, under the same privacy budget.


## How to run
The codebase is split into three distinct folders. Our implementations of basic DP-GAN model, RDPCGAN and PATEGAN. Each folder consists of subfolders containing the specific experiment settings for three datasets - UCI Epileptic, Kaggle Cardiovascular disease and Kaggle Cervical Cancer dataset.

All experiments can be run using the ``` runfiles ```, however some initial fixing of paths related to your individual setup as well as setting up proper libraries may be needed. We have run our experiments on Ubuntu, although other UNIX based system will likely function as well. 

## Additional support

For any additional questions feel free to contact the author @VojtechCrha either here or through my [email](mailto:vojtacrhax@gmail.com).


## Credit
We thank the authors of PATEGAN and RDPCGAN as well as the authors of the used datasets.

#### DP-GANS 
```
 Amirsina Torfi, Edward A. Fox, and Chandan K. Reddy. “Differentially private synthetic
medical data generation using convolutional GANs”. In: Inf. Sci. 586 (2022), pp. 485–
500. DOI: 10.1016/J.INS.2021.12.018. URL: https://doi.org/10.1016/j.ins.
2021.12.018
```
```
James Jordon, Jinsung Yoon, and Mihaela van der Schaar. “PATE-GAN: Generating
Synthetic Data with Differential Privacy Guarantees”. In: 7th International Conference
on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019. Open-
Review.net, 2019. URL: https://openreview.net/forum?id=S1zk9iRqF7
```


#### Datasets
```
 Ralph Andrzejak et al. “Indications of nonlinear deterministic and finite-dimensional struc-
tures in time series of brain electrical activity: Dependence on recording region and brain
state”. In: Physical review. E, Statistical, nonlinear, and soft matter physics 64 (Jan.
2002), p. 061907. DOI: 10.1103/PhysRevE.64.061907
```
```
Svetlana Ulianova. “Cardiovascular Disease Dataset”. In: www.kaggle.com (2019)
```
```
Kelwin Fernandes, Jaime S. Cardoso, and Jessica Fernandes. “Transfer Learning with
Partial Observability Applied to Cervical Cancer Screening”. In: Pattern Recognition and
Image Analysis - 8th Iberian Conference, IbPRIA 2017, Faro, Portugal, June 20-23,
2017, Proceedings. Ed. by Luís A. Alexandre, José Salvador Sánchez, and João M. F.
Rodrigues. Vol. 10255. Lecture Notes in Computer Science. Springer, 2017, pp. 243–
250. DOI: 10.1007/978-3-319-58838-4\_27. URL: https://doi.org/10.1007/978-
3-319-58838-4%5C_27
```