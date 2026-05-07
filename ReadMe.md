# SegMangalFormer

<p align="center">
    <img src="docs/images.png" alt="overview" width="20%" />
</p>

Mangroves are ecosystems subject to tidal disturbance and possess unique vertical structures. Traditional studies of mangrove structure often face challenges due to subtle differences in canopy structure and tidal levels. To address this issue, we propose a SegMangalFormer to perform segmentation and biophysical parameter estimation in complex mangrove structures.


## Dataset

We construct a Mangal-ins dataset that contains over 40 million manually annotated points.



## Traning

After preparing the preprocessed point clouds, put it in the data folder. Then simply run the following code

```
python train.py
```

