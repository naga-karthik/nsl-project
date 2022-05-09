### Neural Scaling Laws Project

This repository contains the code for our project in the course [IFT6760B: Neural Scaling Laws and Foundation Models](https://sites.google.com/view/nsl-course/) taught in the Winter semester of 2022 at Mila/University of Montreal. 

The slides for our presentation, titled "On Layer Normalization for Vision Transformers", can be found [here](https://docs.google.com/presentation/d/1C4SO6YdotOfD7Q2PRqTEP2n73zIYqAcU8t2aqabKaOE/edit?pli=1#slide=id.p).

The goal of our project was to understand the effect of PreNorm and PostNorm versions of the [Vision Transformer](https://arxiv.org/pdf/2010.11929.pdf). While the literature contains some studies that look into the Pre- and Post-Norm versions of the _[vanilla transformer](https://papers.nips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)_ applied to language data, a similar analysis for vision data using vision transformers (ViT) is lacking. We used 4 datasets: CIFAR10, CIFAR100, Imagenette, and Imagewoof and trained them from scratch. 

The idea for this project was primarily inspired by [this](https://arxiv.org/pdf/2002.04745.pdf) paper.
