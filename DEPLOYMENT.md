# 🚀 Deployment Guide - BotTrainer

This guide explains how to host your **BotTrainer** application for free on the web using **Streamlit Community Cloud**.

---

## ☁️ Option 1: Streamlit Community Cloud (Recommended)
This is the easiest and most professional way to share your NLU dashboard with the world.

### 1. Push to GitHub
First, ensure your project is in a public or private GitHub repository.
> [!IMPORTANT]
> Do **NOT** push your `.env` file to GitHub. It contains your private `HF_TOKEN`.

### 2. Connect to Streamlit
1. Go to [share.streamlit.io](https://share.streamlit.io).
2. Click **"New App"** and select your repository, branch (`main`), and main file (`streamlit_app.py`).

### 3. Configure Secrets (The Token)
Since your `.env` file isn't on GitHub, you need to tell Streamlit your token:
1. In the Streamlit app deployment setting, click **"Advanced Settings"**.
2. Go to the **"Secrets"** section.
3. Paste the following:
   ```toml
   HF_TOKEN = "hf_your_token_value_here"
   ```
4. Click **Save**.

### 4. Deploy!
Click **"Deploy!"**. Your app will be live at a custom `share.streamlit.io` URL in about a minute.

---

## 🔧 Troubleshooting Deployment

### ❌ Error: "ModuleNotFoundError"
Ensure your `requirements.txt` is in the root directory and contains all necessary packages:
- `streamlit`
- `huggingface_hub`
- `python-dotenv`
- `pandas`
- `plotly`
- `scikit-learn`
- `pydantic`

### ❌ Error: "403 Forbidden"
This usually means your `HF_TOKEN` in the **Secrets** section is wrong or doesn't have "Inference" permissions. Double-check your [Hugging Face Settings](https://huggingface.co/settings/tokens).

### ⏳ App is slow to load
On the free tier, the Llama-3 model may go to "sleep" if unused. The first request after deployment might take 30 seconds to wake the model up. This is normal!

---

*Congratulations! Your NLU system is now reachable from anywhere in the world!* 🌍
