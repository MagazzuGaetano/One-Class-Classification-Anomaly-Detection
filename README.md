# One-Class-Classification-Anomaly-Detection
A project based on One Class Classification Anomaly Detection to detect defects in hazelnuts like cracks, holes, cuts, ...
This work is based on [DFR](https://github.com/YoungGod/DFR) with the addition of [non-local blocks](https://github.com/tea1528/Non-Local-NN-Pytorch) to capture the spatial context and improve the model anomaly recognition capabilities.

Data used: [MVTecAD](https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads)

Anomaly Classification & Localization:

![image](https://github.com/MagazzuGaetano/One-Class-Classification-Anomaly-Detection/assets/30373288/ae95c66d-8a6f-490c-b176-da6da4bd8451)
![image](https://github.com/MagazzuGaetano/One-Class-Classification-Anomaly-Detection/assets/30373288/77db2996-5fad-42f5-b8e1-3cf9a5707f97)

Prediction examples:

![image](https://github.com/MagazzuGaetano/One-Class-Classification-Anomaly-Detection/assets/30373288/3c183be3-5a11-4484-9e9a-4a22fe782ee7)
![image](https://github.com/MagazzuGaetano/One-Class-Classification-Anomaly-Detection/assets/30373288/84ba81da-3ec1-41d9-bb1d-60a0e424fe6d)
![image](https://github.com/MagazzuGaetano/One-Class-Classification-Anomaly-Detection/assets/30373288/8a8a926c-634a-41aa-b254-04e61a668ae5)
![image](https://github.com/MagazzuGaetano/One-Class-Classification-Anomaly-Detection/assets/30373288/30825ad7-dfa7-451d-a4c4-34096aec7220)
![image](https://github.com/MagazzuGaetano/One-Class-Classification-Anomaly-Detection/assets/30373288/3920baea-2f7a-4b12-a400-27506c69d311)

References:

- Yang, Jie, Yong Shi, and Zhiquan Qi. ["Dfr: Deep feature reconstruction for unsupervised anomaly segmentation."](https://arxiv.org/pdf/2012.07122.pdf) arXiv preprint arXiv:2012.07122 (2020).

- Wang, Xiaolong, et al. ["Non-Local Neural Networks."](https://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Non-Local_Neural_Networks_CVPR_2018_paper.pdf) Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.

- Paul Bergmann, Kilian Batzner, Michael Fauser, David Sattlegger, Carsten Steger: [The MVTec Anomaly Detection Dataset: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection](https://link.springer.com/content/pdf/10.1007/s11263-020-01400-4.pdf); in: International Journal of Computer Vision 129(4):1038-1059, 2021, DOI: 10.1007/s11263-020-01400-4.

- Paul Bergmann, Michael Fauser, David Sattlegger, Carsten Steger: [MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection](https://www.mvtec.com/fileadmin/Redaktion/mvtec.com/company/research/datasets/mvtec_ad.pdf); in: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 9584-9592, 2019, DOI: 10.1109/CVPR.2019.00982.
