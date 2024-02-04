# ai_image_env

A browser interface based on Gradio library for Stable Diffusion, input from .env files for an easy image creation.
![](src/ai_image_env.png)

# Installation:
- METHOD 1:
1. Install [Python 3.10.6](https://www.python.org/downloads/release/python-3106/), checking "Add Python to PATH".
2. Install [git](https://git-scm.com/download/win).
3. On terminal:
```bash
git clone https://github.com/shiroppo/ai_image_env
cd ai_image_env
py -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```
- and then:
```bash
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```
- METHOD 2:
1. Install [Python 3.10.6](https://www.python.org/downloads/release/python-3106/), checking "Add Python to PATH".
2. Install [git](https://git-scm.com/download/win).
3. Download and run [app_install.bat](https://github.com/another-ai/ai_image_env/blob/master/app_install.bat) in the directory of your choice.
# Run
- click on app_gradio.bat on ai_image_env directory
- the interface is now open on the default browser, than select the .env/.yaml file that you prefer(in ./env/ directory)
- select the cycles(how many images you want)
- click "Submit"
- you can find the output in the ./image/_today_date_/ directory
---
- Download from hugginface or any similar website every checkpoint sd xl that you use and put them into ./models/Stable-diffusion/
- Download from hugginface or any similar website every checkpoint sd 1.5/2.0/2.1 that you use and put them into ./models/Stable-diffusion/1.5/
- Download from hugginface or any similar website every LoRA sd xl that you use and put them into ./models/Lora/
- Download from hugginface or any similar website every LoRA sd 1.5/2.0/2.1 that you use and put them into ./models/Lora/1.5/
- Download from hugginface or any similar website every VAE that you use and put them into ./models/VAE/
- Download from hugginface or any similar website every negative embeddings that you use and put them into ./embeddings/
---
# Compel
In the prompt/negative prompt you can use the "prompt weighting":
```bash
- Example: "a grey cat++ playing with a ball----"
```
- in this example the cat(+) is MORE important that the ball(-)

# Files input compatibility:
- .env
- .yaml
- .png(created with ai_image_env)

# Turbo Mode(images ready in few seconds):
(less quality, but more speed)
In the selected .env:
- turbo_lcm=true
- for sd 1.5 checkpoint download the LoRA: https://huggingface.co/cleanmists/my/blob/61484b2d466b8a10349c5207981ea893e74f1c49/LCM_LoRA_Weights_SD15.safetensors and put it into ./models/Lora/1.5/
- for sd xl checkpoint download the LoRA: https://huggingface.co/shiroppo/sd_xl_turbo_lora/blob/main/sd_xl_turbo_lora_v1-128dim.safetensors and put it into ./models/Lora/

# Versions
- v1.0: First Version
- v1.1: Added "Prompt" textbox, if empty the prompt is obtained from the selected *.env file
- v1.2: Added "Turbo Mode"
  
# Support:
- ko-fi: (https://ko-fi.com/shiroppo)
