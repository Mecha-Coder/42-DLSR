### **Feature Selection**


|Features | Pair Plat Matrix |
|---------|------------------|
| Astronomy<br>Herbology<br>Divination<br>Muggle Studies<br>Ancient Runes<br>History of Magic<br>Transfiguration | ![](https://github.com/Mecha-Coder/42-DLSR/blob/jason/figure/feature_selection.png)|

---

### **Gradient Descent Training Results**

| Batch GB | Stochastic GB | Mini-Batch GB |
|----------|---------------|---------------|
| ![](https://github.com/Mecha-Coder/42-DLSR/blob/jason/figure/train_batch_GD.png) | ![](https://github.com/Mecha-Coder/42-DLSR/blob/jason/figure/train_stochastic_GD.png) | ![](https://github.com/Mecha-Coder/42-DLSR/blob/jason/figure/train_mini_batch_GD.png) |

---

### **How to run**

- Note: Plots can be found `./figure` directory

1) Data Exploration
```bash
python3 ./script/histogram.py
python3 ./script/scatter.py
python3 ./script/pair_plot.py
```

2) Different training
```bash
# Batch

# Mini-Batch
python3 ./script/train_bonus_1.py data/dataset_train.csv 

# Stochastic
python3 ./script/train_bonus_2.py data/dataset_train.csv
```

3) Predict
```
python3 script/logreg_predict.py data/dataset_train.csv  logreg.npz
```
