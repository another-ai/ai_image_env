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
- You can modify/create new .env files(.env contains the default settings/base settings)
- Put the sd xl checkpoint that you use in .\models\
- Put the sd 1.5/2.0/2.1 checkpoint that you use in .\models\1.5\
- Put the sd xl LoRA that you use in .\models\Lora\
- Put the sd 1.5/2.0/2.1 LoRA that you use in .\models\Lora\1.5\
- Put the sd VAE that you use in .\models\VAE\
- Put the sd negative embeddings that you use in .\models\embeddings\

# Compel
In the prompt/negative prompt you can use the "prompt weighting":
```bash
- Example: "a red cat++ playing with a ball----"
```
- in this example the cat(+) is MORE important that the ball(-)
