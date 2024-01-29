import os
import os
from PIL import Image
import gradio as gr
from dotenv import load_dotenv

def image_print_create(file_input_name, cycles, prompt):
    if os.path.isfile("./env/merged_file.env"):
        os.remove("./env/merged_file.env")

    try:
        if file_input_name[-4:] == ".env" or file_input_name[-5:] == ".yaml" :
                env_file = file_input_name
                with open("./env/.env", 'r') as file1, open(env_file, 'r') as file2:
                    content1 = file1.read()
                    content2 = file2.read()

                merged_content = content1 + "\n" + content2

                with open("./env/merged_file.env", 'w') as merged_file:
                    merged_file.write(merged_content)
                    import subprocess
                    subprocess.run(["attrib","+H","./env/merged_file.env"],check=True)

                load_dotenv("./env/merged_file.env")
                env_loaded = True
        else:
            image = Image.open(file_input_name)

            metadata = image.info

            if 'env' in metadata:
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

                load_dotenv("./env/merged_file.env")
                env_loaded = True
            else:
                env_file = "./env/.env"
                try:
                    load_dotenv(env_file) 
                except:
                    env_loaded = False
                    print(f"Error! {env_file} not loaded")
                else:
                    env_loaded = True
    except:
        env_file = "./env/.env"
        try:
            load_dotenv(env_file) 
        except:
            env_loaded = False
            print(f"Error! {env_file} not loaded")
        else:
            env_loaded = True

    if prompt == "":
        prompt_input = os.getenv("prompt_input","1girl")
    else:
        prompt_input = prompt
    cycles = cycles
    dynamic_prompt = int(os.getenv("dynamic_prompt", "0"))
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
        inputs=["file",gr.Number(value=1, label="Cycles"),gr.Textbox(value="", lines=4, label="Prompt")],
        outputs=["image"],
        title="Image Create",
        allow_flagging="never"
    )
    interface.launch(share=False, inbrowser=True)