# Plant Leaf Disease Classification, deployed on Streamlit

Image classifiers for mango and tomato leaf diseases, trained on PlantVillage and served in a Streamlit app where a user picks the plant, uploads a leaf photo, and gets the diagnosis.

## Results (held-out test sets)

| Plant | Classes | Images | Model | Accuracy |
|---|---|---|---|---|
| Mango | 8 | ~4,000 | Xception (`Xception.ipynb`) | **98.0%** |
| Mango | 8 | ~4,000 | InceptionV3 (`INCEPTIONMANGO.ipynb`) | 95.3% |
| Tomato | 10 | ~10,000 | Custom CNN (`tomato.ipynb`) | 94.5% (val) |
| Tomato | 10 | ~10,000 | VGG (`vggtom.ipynb`) | 86.6% |

Xception won on mango; on tomato the from-scratch CNN beat VGG transfer learning — the tomato classes are visually closer together and the smaller model overfit less.

## App

`main.py` — Streamlit. Sidebar selects mango or tomato, loads the matching model, runs inference on an uploaded image and shows the predicted disease with confidence.

```bash
pip install streamlit tensorflow opencv-python pillow
streamlit run main.py
```

![app](https://github.com/Mananpatel25/Plant-leaf-disease-classification-on-streamlit/assets/85306973/43a25d94-9bc5-4722-8e61-9d995f4cfab1)
![app](https://github.com/Mananpatel25/Plant-leaf-disease-classification-on-streamlit/assets/85306973/7184dc1f-4419-4f77-8081-45f532908660)

## Data

PlantVillage via Kaggle — mango (8 classes) and tomato (10 classes) subsets. Train/val/test split 70/15/15, standard augmentation (flip, rotate, zoom).
