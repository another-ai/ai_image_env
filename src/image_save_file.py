import os
from datetime import datetime as date_time
import re
from PIL import Image, ImageEnhance
from PIL.PngImagePlugin import PngInfo

def count_file(directory_path_temp):
    unique_id_temp = 0
    existing_files = len([f for f in os.listdir(directory_path_temp) if (f.endswith(".png") or f.endswith(".jpg")) and (os.path.isfile(os.path.join(directory_path_temp, f)))])
    unique_id_temp = existing_files + 1
    return unique_id_temp

def get_highest_folder_name(directory):
    folders = [f for f in os.listdir(directory) if os.path.isdir(f)]

    folders = [f for f in folders if f.isdigit()]

    folders.sort(key=int)

    if not folders:
        return str(1)
    else:
        highest = folders[-1]
        return str(highest)

def count_folders(directory_path_temp, new_folder):
    unique_id_temp = 0
    existing_folders = [
        int(d.split('_')[0]) for d in os.listdir(directory_path_temp) if (os.path.isdir(os.path.join(directory_path_temp, d)) and re.search(r'^\d+', d))
    ]
    if existing_folders:
        unique_id_temp = max(existing_folders)
        if new_folder:
            unique_id_temp = unique_id_temp + 1    
    else:
        unique_id_temp = 1
    return str(unique_id_temp)

def add_metadata_file(file_path, txt_file_data_file, env_file_text):
    targetImage = Image.open(file_path)
    metadata = PngInfo()
    metadata.add_text("parameters", txt_file_data_file)
    metadata.add_text("env", env_file_text)
    targetImage.save(file_path, pnginfo=metadata)

def constrast_image(image_file, factor):
    im_constrast = ImageEnhance.Contrast(image_file).enhance(factor)
    return im_constrast

def save_file(image_file, txt_file_data_file, new_folder, create_story, prompt="", env_file="", env_file_text=""):
    if env_file == "" or env_file[-5:] == "/.env" or env_file[-5:] == "\.env":
        env_file_name = ""
    else:
        env_file_name = os.path.basename(os.path.splitext(env_file)[0])

    file_path = ""
    if image_file != "":
        current_datetime = date_time.now()
        current_date = current_datetime.strftime("%Y_%m_%d")
        current_time = current_datetime.strftime("%H_%M_%S")
        if not os.path.exists("./image"):
            os.makedirs("./image")
        if not os.path.exists("./image/" + current_date):
            os.makedirs("./image/" + current_date)
        unique_id_folders = count_folders("./image/" + current_date, new_folder)
        if env_file_name != "":
            directory_path = f"./image/{current_date}/{unique_id_folders}_{env_file_name}"
        else:
            directory_path = f"./image/{current_date}/{unique_id_folders}"
        print(f"Directory:{directory_path}")
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        os.makedirs(directory_path, exist_ok=True)
        unique_id = count_file(directory_path)
        file_name = f"{unique_id}_{current_time}.png"
        file_path = f"{directory_path}/{file_name}"
        image_file = constrast_image(image_file, 1.1)
        image_file.save(file_path)
        add_metadata_file(file_path, txt_file_data_file, env_file_text)
    return file_path

def save_file_alt(image_file, txt_file_data_file, directory_save,env_file_text):
    file_path = ""
    if image_file != "":
        current_datetime = date_time.now()
        current_date = current_datetime.strftime("%Y_%m_%d")
        current_time = current_datetime.strftime("%H_%M_%S")
        if not os.path.exists("./image"):
            os.makedirs("./image")
        if not os.path.exists("./image/" + current_date):
            os.makedirs("./image/" + current_date)
        if not os.path.exists("./image/" + current_date + "/" + directory_save):
            os.makedirs("./image/" + current_date + "/" + directory_save)
        directory_path = f"./image/{current_date}/{directory_save}"
        print(f"Directory:{directory_path}")
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        os.makedirs(directory_path, exist_ok=True)
        unique_id = count_file(directory_path)
        file_name = f"{unique_id}_{current_time}.png"
        file_path = f"{directory_path}/{file_name}"
        image_file = constrast_image(image_file, 1.1)
        image_file.save(file_path)
        add_metadata_file(file_path, txt_file_data_file, env_file_text)
    return file_path

if __name__ == "__main__":
    save_file("", "")
