# One-Class-Classification-Anomaly-Detection
A project based on One Class Classification Anomaly Detection to detect defects in hazelnuts like cracks, holes, cuts, ...
This work is based on [DFR](https://github.com/YoungGod/DFR) with the addition of [non-local blocks](https://github.com/tea1528/Non-Local-NN-Pytorch) to capture the spatial context and improve the model anomaly recognition capabilities.

Data used: [MVTecAD](https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads)

Anomaly Classification & Localization Results:
|                                                  **Models**                                                  | **AUC-ROC** | **Pixel AUC-ROC** |
|:------------------------------------------------------------------------------------------------------------:|:-----------:|:-----------------:|
|                                    [DFR](https://github.com/YoungGod/DFR)                                    |    0.9950   |       0.9826      |
| [DFR](https://github.com/YoungGod/DFR) + [non-local blocks](https://github.com/tea1528/Non-Local-NN-Pytorch) |    0.9975   |       0.9836      |


Prediction examples:
![good](https://github.com/MagazzuGaetano/One-Class-Classification-Anomaly-Detection/assets/30373288/9599c5f8-c866-4cf4-9fd2-4e36df008fe9)
![cut](https://github.com/MagazzuGaetano/One-Class-Classification-Anomaly-Detection/assets/30373288/be68735e-77d3-44b5-aa92-6e735429ae6b)
![hole](https://github.com/MagazzuGaetano/One-Class-Classification-Anomaly-Detection/assets/30373288/240f8480-2b88-43bc-9ded-01fe0d54a48e)
![crack](https://github.com/MagazzuGaetano/One-Class-Classification-Anomaly-Detection/assets/30373288/3281414e-24a6-4709-bce9-70647776b480)
![print](https://github.com/MagazzuGaetano/One-Class-Classification-Anomaly-Detection/assets/30373288/19e68d03-f541-49e6-acf7-2664fd8f4599)

References:


- Yang, Jie, Yong Shi, and Zhiquan Qi. ["Dfr: Deep feature reconstruction for unsupervised anomaly segmentation."](https://arxiv.org/pdf/2012.07122.pdf) arXiv preprint arXiv:2012.07122 (2020).

- Wang, Xiaolong, et al. ["Non-Local Neural Networks."](https://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Non-Local_Neural_Networks_CVPR_2018_paper.pdf) Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.

- Paul Bergmann, Kilian Batzner, Michael Fauser, David Sattlegger, Carsten Steger: [The MVTec Anomaly Detection Dataset: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection](https://link.springer.com/content/pdf/10.1007/s11263-020-01400-4.pdf); in: International Journal of Computer Vision 129(4):1038-1059, 2021, DOI: 10.1007/s11263-020-01400-4.

- Paul Bergmann, Michael Fauser, David Sattlegger, Carsten Steger: [MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection](https://www.mvtec.com/fileadmin/Redaktion/mvtec.com/company/research/datasets/mvtec_ad.pdf); in: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 9584-9592, 2019, DOI: 10.1109/CVPR.2019.00982.
