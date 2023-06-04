# ABC - Batch correction of single cell sequencing data via an autoencoder architecture

ABC (Autoencoder-based Batch Correction) is a semi-supervised deep learning architecture for integrating
single cell sequencing datasets. This method removes batch effects through a guided process of data compression
using supervised cell type classifier branches for biological signal retention. It aligns the different batches
using an adversarial training approach.


## Installation 

```bash
pip install git+https://github.com/reutd/ABC.git

```

## Usage
The model takes an anndata object and two string variables: batch_label and type_label representing the batch label and cell type label keys into anndata.obs.
Please make sure to use preprocessed data after normalization and log1 transformation. (for example using the scanpy methods sc.pp.normalize_total and sc.pp.log1p).

First, create the ABC model:
```bash
model = ABC(adata, batch_key, label_key)
```
then remove batch effects using:
```bash
integrated = model.batch_correction()
```
integrated will hold a new anndata object with the corrected values.


## Example
An integration example of a Lung Atlas dataset can be found under the Example directory.
The Lung Atlas dataset in the example is published by
Luecken, M.D., Büttner, M., Chaichoompu, K. et al. Benchmarking atlas-level data integration in single-cell genomics. Nat Methods 19, 41–50 (2022). https://doi.org/10.1038/s41592-021-01336-8
and is publicly available as Anndata objects on Figshare at: https://doi.org/10.6084/m9.figshare.12420968.


