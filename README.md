# **Rendezvous in Time**: An Attention-based Temporal Fusion approach for Surgical Triplet Recognition (IPCAI 2023)

<i>S. Sharma, C. I. Nwoye, D. Mutter, N. Padoy</i>

This is an official pytorch implementation of paper Rendezvous in Time (RiT). 

[![arXiv](https://img.shields.io/badge/arXiv-2211.16963-f9f107.svg?style=flat)](https://arxiv.org/abs/2211.16963)

# Abstract
One of the recent advances in surgical AI is the recognition of surgical activities as triplets of (instrument, verb, target). Albeit providing detailed information for computer-assisted intervention, current triplet recognition approaches rely only on single frame features. Exploiting the temporal cues from earlier frames would improve the recognition of surgical action triplets from videos. 
In this paper, we propose Rendezvous in Time (RiT) - a deep learning model that extends the state-of-the-art model, Rendezvous, with temporal modeling. Focusing more on the verbs, our RiT explores the connectedness of current and past frames to learn temporal attention-based features for enhanced triplet recognition. We validate our proposal on the challenging surgical triplet dataset, CholecT45, demonstrating an improved recognition of the verb and triplet along with other interactions involving the verb such as (instrument, verb). Qualitative results show that the RiT produces smoother predictions for most triplet instances than the state-of-the-arts. We present a novel attention-based approach that leverages the temporal fusion of video frames to model the evolution of surgical actions and exploit their benefits for surgical triplet recognition. 

# Model Overview
![Rendezvous in Time](images/rit_model.jpg)


# Model Description & Weights [Coming Soon...]


If you find RiT useful in your research, please use the following BibTeX entry for citation.

```BibTeX
@article{sharma2022rendezvous,
  title={Rendezvous in Time: An Attention-based Temporal Fusion approach for Surgical Triplet Recognition},
  author={Sharma, Saurav and Nwoye, Chinedu Innocent and Mutter, Didier and Padoy, Nicolas},
  journal={arXiv preprint arXiv:2211.16963},
  year={2022}
}
```


