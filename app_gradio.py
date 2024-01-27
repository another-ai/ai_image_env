import os
import os
from PIL import Image
import gradio as gr
from dotenv import load_dotenv

def image_print_create(file_input_name, cycles):
    if os.path.isfile("./env/merged_file.env"):
        os.remove("./env/merged_file.env")

    try:
        if file_input_name[-4:] == ".env":
                env_file = file_input_name
                with open("./env/.env", 'r') as file1, open(env_file, 'r') as file2:
                    content1 = file1.read()
                    content2 = file2.read()

                merged_content = content1 + "\n" + content2

                with open("./env/merged_file.env", 'w') as merged_file:
                    merged_file.write(merged_content)
                    import subprocess
                    subprocess.run(["attrib","+H","./env/merged_file.env"],check=True)

                load_dotenv("./env/merged_file.env") # env_file overwrite ./env/.env
                env_loaded = True
                print("merged_file.env loaded")   
        else:
            # Carica l'immagine PNG
            image = Image.open(file_input_name)

            # Ottieni il metadata dell'immagine
            metadata = image.info

            # Verifica se il parametro 'env' è presente nei metadata
            if 'env' in metadata:
                # Leggi il contenuto del parametro 'env'
                env_content = metadata['env']
                env_file = "file"

                with open("./env/merged_file.env", 'w') as merged_file:
                    merged_file.write(env_content)

                with open("./env/.env", 'r') as file1, open("./env/merged_file.env", 'r') as file2:
                    content1 = file1.read()
                    content2 = file2.read()

                merged_content = content1 + "\n" + content2

                with open("./env/merged_file.env", 'w') as merged_file:
                    merged_file.write(merged_content)
                    import subprocess
                    subprocess.run(["attrib","+H","./env/merged_file.env"],check=True)

                load_dotenv("./env/merged_file.env") # env_file overwrite ./env/.env
                env_loaded = True
                print("merged_file.env loaded")
            else:
                env_file = "./env/.env"
                try:
                    load_dotenv(env_file) 
                except:
                    env_loaded = False
                    print(f"Error! {env_file} not loaded")
                else:
                    env_loaded = True
                    print(f"{env_file} loaded")
    except:
        env_file = "./env/.env"
        try:
            load_dotenv(env_file) 
        except:
            env_loaded = False
            print(f"Error! {env_file} not loaded")
        else:
            env_loaded = True
            print(f"{env_file} loaded")

    prompt_input = os.getenv("prompt_input","1girl")
    cycles = cycles # int(os.getenv("cycles", "10"))
    dynamic_prompt = int(os.getenv("dynamic_prompt", "0")) # number means max new token, 64 = default, 0 = off
    directory_save = ""
    main_dir=os.getenv("main_dir","./")
    if main_dir[-1] != "/":
        main_dir = main_dir + "/"
    from app import main_def
    image = main_def(env_loaded=env_loaded,env_file=env_file,main_dir=main_dir,prompt_input=prompt_input,cycles=cycles,dynamic_prompt=dynamic_prompt,directory_save=directory_save)
    if os.path.isfile("./env/merged_file.env"):
        os.remove("./env/merged_file.env")
    return image

if __name__ == "__main__":

    interface = gr.Interface(
        fn=image_print_create,
        inputs=["file",gr.Number(value=1, label="Cycles")],
        outputs=["image"],
        title="Image Create", # description="SD-Easy",
        allow_flagging="never"
        # live=True
    )
    interface.launch()