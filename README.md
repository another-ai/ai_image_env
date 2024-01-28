# ai_image_env

A browser interface based on Gradio library for Stable Diffusion, input from .env files for an easy image creation.

# Installation:

```bash
git clone https://github.com/shiroppo/ai_image_env
cd ai_image_env
py -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```
# Run:
```bash
.\venv\Scripts\activate
py app_gradio.py
```
- You can modify/create new .env file and than put them into the env dir.
- Download from hugginface or any similar website every checkpoint than you use and put them into ./models/ if they are stable diffusion xl models or ./models/1.5 if they are stable diffusion 1.5/2.0/2.1 models
- Download from hugginface or any similar website every LoRA than you use and put them into ./models/LoRA if they are stable diffusion xl LoRA or ./models/1.5 if they are stable diffusion 1.5/2.0/2.1 LoRA
- Download from hugginface or any similar website every negative embedding than you use and put them into ./embeddings/
