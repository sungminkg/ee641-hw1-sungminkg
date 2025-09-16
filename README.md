# EE641 Homework 1

**Full Name:** Sungmin Kang  
**USC Email:** kangsung@usc.edu  

---

## Instructions

### Problem 1: Multi-Scale Single-Shot Detector
To train and evaluate the object detection model:

```bash
cd problem1
python train.py
python evaluate.py
```


### Problem 2: Heatmap vs Direct Regression for Keypoint Detection
To train and evaluate the object detection model:

```bash
cd problem2
python train.py
python evaluate.py
python baseline.py
```

---

## Implementation Notes

- Problem 1: Simplified SSD with feature pyramid, outputs detection results and mAP scores.
- Problem 2: Comparison of HeatmapNet vs RegressionNet, includes PCK evaluation, ablation, and failure case analysis.
- Results, models, and visualizations are saved under each problem’s results/ directory.


Folder Tree: 

ee641-hw1-sungminkg/
├── problem1/
│   ├── model.py
│   ├── dataset.py
│   ├── loss.py
│   ├── train.py
│   ├── evaluate.py
│   ├── utils.py
│   └── results/
│       ├── training_log.json
│       ├── best_model.pth
│       └── visualizations/
├── problem2/
│   ├── model.py
│   ├── dataset.py
│   ├── train.py
│   ├── evaluate.py
│   ├── baseline.py
│   └── results/
│       ├── training_log.json
│       ├── heatmap_model.pth
│       ├── regression_model.pth
│       └── visualizations/
├── report.pdf
└── README.md
