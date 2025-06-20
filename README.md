# Image Classification & Explainability

This repository contains XXX

---

## 1. Clone the repository

```bash
git clone https://github.com/MatteoDeMoor/Image_Classification_And_Explainability
cd Image_Classification_And_Explainability
```

---

## 2. Set up a Python virtual environment

Create and activate a new venv in your project folder:

```bash
python -m venv venv
```

```powershell
.\venv\Scripts\Activate.ps1
```

---

## 3. Install & lock your dependencies

1. Upgrade pip and install **pip-tools**:  
   ```bash
   pip install --upgrade pip
   pip install pip-tools
   ```
2. Compile your lockfile (`requirements.txt`) from your top-level specs (`requirements.in`):  
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

## 4. XXX

---

## 5. Launch the services with Docker Compose

Build and start both the FastAPI backend and the Gradio UI in separate containers:

```bash
docker-compose up --build
```

---

## 6. Version control & workflow

- **Commit & push** your changes:
  ```bash
  git status
  git add .
  git commit -m "Your message"
  git push
  ```
- **.gitignore** should include:
  ```
  venv/
  ```

---

## Benchmark Results



## License

```text
Copyright (c) 2025 Matteo De Moor
All Rights Reserved.

No permission is granted to copy, modify or redistribute this software without explicit written consent from the author.
```
