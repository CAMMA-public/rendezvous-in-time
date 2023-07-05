# **Rendezvous in Time**: An Attention-based Temporal Fusion approach for Surgical Triplet Recognition (IPCAI 2023)

<i>S. Sharma, C. I. Nwoye, D. Mutter, N. Padoy</i>

This is an official implementation of Rendezvous in Time (RiT) in PyTorch. Based on [Rendezvous](https://github.com/CAMMA-public/rendezvous) PyTorch codebase.

[![arXiv](https://img.shields.io/badge/arXiv-2211.16963-f9f107.svg?style=flat)](https://arxiv.org/abs/2211.16963)
[![Journal Publication](https://img.shields.io/badge/Spinger-IJCARS-magenta)](https://link.springer.com/article/10.1007/s11548-023-02914-1)

## News
* [04/07/2023] Release of code in PyTorch for training and evaluation.

## Abstract
One of the recent advances in surgical AI is the recognition of surgical activities as triplets of (instrument, verb, target). Albeit providing detailed information for computer-assisted intervention, current triplet recognition approaches rely only on single frame features. Exploiting the temporal cues from earlier frames would improve the recognition of surgical action triplets from videos. 
In this paper, we propose Rendezvous in Time (RiT) - a deep learning model that extends the state-of-the-art model, Rendezvous, with temporal modeling. Focusing more on the verbs, our RiT explores the connectedness of current and past frames to learn temporal attention-based features for enhanced triplet recognition. We validate our proposal on the challenging surgical triplet dataset, CholecT45, demonstrating an improved recognition of the verb and triplet along with other interactions involving the verb such as (instrument, verb). Qualitative results show that the RiT produces smoother predictions for most triplet instances than the state-of-the-arts. We present a novel attention-based approach that leverages the temporal fusion of video frames to model the evolution of surgical actions and exploit their benefits for surgical triplet recognition. 

## Model Overview
![Rendezvous in Time](images/rit_model.jpg)

<br>

## Pre-requisities
* Create conda environment and install packages from `requirements.txt`.
* Download the CholecT50 dataset from https://github.com/CAMMA-public/cholect50.
* Install [ivtmetrics](https://github.com/CAMMA-public/ivtmetrics) for evaluation.
* For more details on the splits, please refer the paper [Data Splits and Metrics](https://arxiv.org/abs/2204.05235).

<br>

## Training Details

RiT has been trained on `Nvidia V100 GPU` with `CUDA version 10.2`. Run the script below to launch training. Currently, the model is adapted to train only on one GPU.
```bash
bash train.sh
```

<br>

## Evaluation
To evaluate, please provide the checkpoint name in `ckp_name` and folder in `ckp_folder` in the argument parser.
```bash 
bash test.sh
```

NOTE: The split corresponding to Table 1 in the paper is `cholect45-crossval`

<br>

## Acknowledgements
This work was supported by French state funds managed by the ANR within the National AI
Chair program under Grant ANR-20-CHIA-0029-01 (Chair AI4ORSafety) and within the Investments for the future
program under Grant ANR-10-IAHU-02 (IHU Strasbourg). It was granted access to the HPC resources of Unistra
Mesocentre.

<br>

## License
This code, models, and datasets are available for non-commercial scientific research purposes as defined in the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). By downloading and using this code you agree to the terms in the [LICENSE](LICENSE). Third-party codes are subject to their respective licenses.


## References
If you find RiT useful in your research, please use the following BibTeX entry for citation.

```bibtex
@article{sharma2023rendezvous,
  title={Rendezvous in time: an attention-based temporal fusion approach for surgical triplet recognition},
  author={Sharma, Saurav and Nwoye, Chinedu Innocent and Mutter, Didier and Padoy, Nicolas},
  journal={International Journal of Computer Assisted Radiology and Surgery},
  pages={1--7},
  year={2023},
  publisher={Springer}
}
```


