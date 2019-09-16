# SDM (Statistical Disease Mapping)

The aim of this package is to propose a novel statistical disease mapping (SDM) framework to address
some abnormal pattern detection challenges. We develop an efficient estimation method to estimate unknown
parameters in SDM and delineate individual and group disease maps.

There are three components of statistical methods included in our SDM: (i)
MVCM, (ii) HMRFM, and (iii) DRM. First, in MVCM, the relationship between
imaging signals and covariates of interest is investigated via the functional data
analysis tools. Compared to the voxel based method, MVCM can not only well
preserve the spatial smoothness and correlation within the imaging signals, but
also model the local heterogeneity among multiple imaging features; Second,
in clinical, the potential diseased regions vary across subjects in terms of their
number, size, and location. To capture this global heterogeneity, the HMRFM
is adopted where the diseased regions are modeled via discrete latent variables;
Finally, it is of great meaning in clinical practice to derive the statistical disease
mapping for certain patient group (e.g., female MCI patients with age range between 60 and 65). The proposed DRM can successfully integrate the individual
diseased region information to form the statistical disease mapping, where the
probability that each pixel belongs to the diseased region is modeled for certain
patient group of interest.

In particular, our SDM package can handle
three types of functional phenotypes including curves, surfaces, and volumes.

## sdm_hippo_s1.py
Run SDM (statistical disease mapping) pipeline on ADNI hippocampal surface data (step 1)
```
Usage: python ./sdm_hippo_s1.py ./data/ ./result/ left
```

## sdm_hippo_s2.py
Run SDM (statistical disease mapping) pipeline on ADNI hippocampal surface data (step 2)
```
Usage: python ./sdm_hippo_s2.py ./data/ ./result/left/ left 10 5
```

## sdm_hippo_s3.R
Run SDM (statistical disease mapping) pipeline on ADNI hippocampal surface data (step 3)
