# Image Classification & Explainability

This repository contains a reproducible pipeline for training a deep CNN on the PlantVillage dataset (sourced from [spMohanty/PlantVillage-Dataset](https://github.com/spMohanty/PlantVillage-Dataset/tree/master/raw/color)) to detect plant diseases, generating Grad-CAM explainability heatmaps, and deploying via Docker with a simple Gradio UI.

---

## 1. Clone the repository

```bash
git clone https://github.com/MatteoDeMoor/Image_Classification_And_Explainability
cd Image_Classification_And_Explainability
```

---

## 2. Download the PlantVillage dataset

We use the original color images from the [PlantVillage-Dataset raw/color folder](https://github.com/spMohanty/PlantVillage-Dataset/tree/master/raw/color):

```bash
git clone https://github.com/spMohanty/PlantVillage-Dataset
cd PlantVillage-Dataset/raw/color
# Copy into project
mkdir -p ../../../../data/PlantVillage
cp -r . ../../../../data/PlantVillage
cd ../../../../
```

**Citation**:  
If you use this dataset, please cite:  
```text
@article{Mohanty_Hughes_Salathé_2016,
  title={Using deep learning for image-based plant disease detection},
  volume={7},
  DOI={10.3389/fpls.2016.01419},
  journal={Frontiers in Plant Science},
  author={Mohanty, Sharada P. and Hughes, David P. and Salathé, Marcel},
  year={2016},
  month={Sep}
}
```

---

## 3. Set up a Python virtual environment

Create and activate a new venv in your project folder:

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
```

---

## 4. Install & lock your dependencies

1. Upgrade pip and install **pip-tools**:  
   ```bash
   pip install --upgrade pip
   pip install pip-tools
   ```
2. Compile your lockfile (`requirements.txt`) from `requirements.in`:  
   ```bash
   pip-compile requirements.in --output-file=requirements.txt
   ```
3. Install all dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
4. (Optional) **Sync** your venv so that only those packages in `requirements.txt` remain:  
   ```bash
   pip-sync requirements.txt
   ```

---

## 5. Run training

Train the CNN and save the best model to `models/best_model.pth`:

```bash
python train.py
```

---

## 6. Launch the Gradio demo

Start the Gradio UI to test inference and view Grad-CAM:

```bash
docker-compose up --build
```

- Open **http://localhost:7860** in your browser.

Use **Ctrl+C** to stop the service.

---

## 7. Version control & workflow

- **Commit & push** your changes:
  ```bash
  git add .
  git commit -m "Your message"
  git push
  ```
- Ensure `data/`, `venv/`, and `models/` are in your `.gitignore`.

---

## License

```text
Copyright (c) 2025 Matteo De Moor
All Rights Reserved.

No permission is granted to copy, modify or redistribute this software without explicit written consent from the author.
```
