# ECOL - Fake News Detection Framework for the Healthcare Domain 
![framework](images/FakeNewsDetectionFramework.png) More details are given in [our paper](https://arxiv.org/abs/2101.05499).

`scripts` folder has examples for reproducing the study:

* `scripts/elasticsearch_ops.sh` is used for extracting relevant past fake news
* `scripts/links_ops.sh` is used for extracting link information on the posts
* `scripts/train_server.sh` is used for training the components of the ECOL

Trained models can be downloaded from [this link](https://www.dropbox.com/sh/yn078rcvv6zfgia/AADT4rVCpYxEtLO46JETj4mTa?dl=0).

# Datasets
The list of additional datasets for training/testing some of the components of the framework:
[FakeHealth](https://zenodo.org/record/3862989)
[NELA-2019](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/O7FWPO)

# Citation
Please use the following citation if you use our code:

`@misc{baris2021ecol,
      title={ECOL: Early Detection of COVID Lies Using Content, Prior Knowledge and Source Information}, 
      author={Ipek Baris and Zeyd Boukhers},
      year={2021},
      eprint={2101.05499},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}`