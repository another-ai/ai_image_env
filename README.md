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
# Run
```bash
.\venv\Scripts\activate
py app_gradio.py
```

- Download from hugginface or any similar website every checkpoint that you use and put them into ./models/ if they are stable diffusion xl models or ./models/1.5/ if they are stable diffusion 1.5/2.0/2.1 models
- Download from hugginface or any similar website every LoRA that you use and put them into ./models/LoRA if they are stable diffusion xl LoRA or ./models/1.5/ if they are stable diffusion 1.5/2.0/2.1 LoRA
- Download from hugginface or any similar website every VAE that you use and put them into ./models/VAE/
- Download from hugginface or any similar website every negative embeddings that you use and put them into ./embeddings/

# Compel
In the prompt/negative prompt you can use the "prompt weighting":
```bash
- Example: "a grey cat++ playing with a ball----"
```
- in this example the cat(+) is MORE important that the ball(-)
