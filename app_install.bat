git clone https://github.com/shiroppo/ai_image_env
cd ai_image_env
py -m venv venv
call .\venv\Scripts\activate
pip install -r requirements.txt
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118