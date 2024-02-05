import sys
import gc
import os
path = os.path.abspath("src")
sys.path.append(path)
from PIL import Image
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, AutoencoderKL
# from diffusers import DDIMScheduler
from diffusers import EulerAncestralDiscreteScheduler # Euler a	EulerAncestralDiscreteScheduler	init with use_karras_sigmas=True from https://huggingface.co/docs/diffusers/main/en/api/schedulers/overview
from diffusers import DPMSolverMultistepScheduler # DPM++ 2M Karras	DPMSolverMultistepScheduler	init with use_karras_sigmas=True from https://huggingface.co/docs/diffusers/main/en/api/schedulers/overview
from diffusers import DPMSolverSinglestepScheduler # DPM++ SDE Karras	DPMSolverSinglestepScheduler init with use_karras_sigmas=True from https://huggingface.co/docs/diffusers/main/en/api/schedulers/overview
from diffusers import LCMScheduler
import torch
import random
import image_save_file
import create_gif
import app_retnet
# import app_audio
# from huggingface_hub import login
import time
import hashlib
# import transformers
from dotenv import load_dotenv
import re
from wildcard_scene import wildcard_scene_def as random_wild

def are_all_not_black(images):
    for image in images:
        if is_completely_black(image):
            print("Error in image creation")
            return False
    return True


def is_completely_black(image):

    for pixel in image.getdata():
        if pixel != (0, 0, 0):
            return False
    return True

def calculate_sha256(filename, cut=10): # for everything except for LoRA
    hash_sha256 = hashlib.sha256()
    blksize = 1024 * 1024

    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(blksize), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()[:cut]

def addnet_hash_safetensors(b, cut=12): # for LoRA
    """kohya-ss hash for safetensors from https://github.com/kohya-ss/sd-scripts/blob/main/library/train_util.py"""
    hash_sha256 = hashlib.sha256()
    blksize = 1024 * 1024

    b.seek(0)
    header = b.read(8)
    n = int.from_bytes(header, "little")

    offset = n + 8
    b.seek(offset)
    for chunk in iter(lambda: b.read(blksize), b""):
        hash_sha256.update(chunk)

    return hash_sha256.hexdigest()[:cut]

def get_pipeline_embeds(pipeline, prompt, negative_prompt, device, truncation_option):
    max_length = pipeline.tokenizer.model_max_length

    count_prompt = len(prompt.split(","))
    count_negative_prompt = len(negative_prompt.split(","))

    if count_prompt >= count_negative_prompt:
        input_ids = pipeline.tokenizer(prompt, return_tensors="pt", truncation=truncation_option).input_ids.to(device)
        shape_max_length = input_ids.shape[-1]
        negative_ids = pipeline.tokenizer(negative_prompt, truncation=truncation_option, padding="max_length",
                                          max_length=shape_max_length, return_tensors="pt").input_ids.to(device)

    else:
        negative_ids = pipeline.tokenizer(negative_prompt, return_tensors="pt", truncation=truncation_option).input_ids.to(device)
        shape_max_length = negative_ids.shape[-1]
        input_ids = pipeline.tokenizer(prompt, return_tensors="pt", truncation=truncation_option, padding="max_length",
                                       max_length=shape_max_length).input_ids.to(device)

    concat_embeds = []
    neg_embeds = []
    for i in range(0, shape_max_length, max_length):
        concat_embeds.append(pipeline.text_encoder(input_ids[:, i: i + max_length])[0])
        neg_embeds.append(pipeline.text_encoder(negative_ids[:, i: i + max_length])[0])

    return torch.cat(concat_embeds, dim=1), torch.cat(neg_embeds, dim=1)

"""
def random_lora(folder, random_lora_enabled):
    if random_lora_enabled:
        safetensors_files = [file for file in os.listdir(folder) if file.endswith(".safetensors")]

        if safetensors_files:
            chosen_file = random.choice(safetensors_files)
            return chosen_file
    return ""
"""

def image_print(env_loaded, env_file, main_dir, prompt_action, prompt_input, negative_prompt, sd_xl, clip_skip, turbo_lcm, no_vae, dynamic_prompt, random_seed, input_seed, cycles, truncation_option, gif_creation, create_story, best_prompt_search, random_dimension, directory_save):
    def remove_last_comma(sentence):
        if len(sentence) > 0 and sentence[-1] == ',':
            sentence_without_comma = sentence[:-1]
            return sentence_without_comma
        else:
            return sentence
        
    def remove_duplicates(words):
        words_list = words.split(",")

        unique_words = []

        for word in words_list:
            if word not in unique_words:
                unique_words.append(word)

        unique_string = ",".join(unique_words)

        return unique_string

    if not env_loaded:
        try:
            load_dotenv("./env/.env")
        except:
            env_loaded = False
            print("Error! ./env/.env not loaded")
        else:
            env_loaded = True
            print("./env/.env loaded")
    
    cycles = int(cycles) # necessary for right external value call
    device_ = "cuda"
    if device_ == "cuda":
        torch.cuda.empty_cache()
    if sd_xl:
        clip_skip = 1
    else:
        clip_skip = int(clip_skip)

    if device_ == "cpu":
        torch_dtype_=torch.float32
    elif sd_xl:
        torch_dtype_=torch.float16 # sd_xl and float32 are very slow
    else:
        torch_dtype_=torch.float16 # float16 causes random black images sometimes

    # login()d
    # pipeline = StableDiffusionPipeline.from_pretrained("hogiahien/aom3", torch_dtype=torch_dtype_, trust_remote_code=True)
        
    sd1_5_dir = os.getenv("sd1_5_dir", "false").lower() == "true"
    if sd_xl:
        checkpoint = os.getenv("checkpoint_xl", "juggernautXL_v7FP16VAEFix.safetensors")
        pipeline = StableDiffusionXLPipeline.from_single_file(main_dir + "Stable-diffusion/" + checkpoint, torch_dtype=torch_dtype_, add_watermarker=False)
        model_hash = calculate_sha256(main_dir + "Stable-diffusion/" + checkpoint, 10)
    else:
        checkpoint = os.getenv("checkpoint", "absolutereality_v181.safetensors")
        if sd1_5_dir:
            pipeline = StableDiffusionPipeline.from_single_file(main_dir + "Stable-diffusion/1.5/" + checkpoint, torch_dtype=torch_dtype_, add_watermarker=False)
            model_hash=calculate_sha256(main_dir + "Stable-diffusion/1.5/" + checkpoint, 10)
        else:
            pipeline = StableDiffusionPipeline.from_single_file(main_dir + "Stable-diffusion/" + checkpoint, torch_dtype=torch_dtype_, add_watermarker=False)           
            model_hash=calculate_sha256(main_dir + "Stable-diffusion/" + checkpoint, 10)

    if no_vae:
        vae_name = "" # for checkpoint with vae already baked in
    elif sd_xl: #realistic and anime
        vae_name = os.getenv("vae_name_xl", "sdxl_vae.safetensors")
    else:
        vae_name = os.getenv("vae_name", "vae-ft-mse-840000-ema-pruned.ckpt") # for reality

    if vae_name != "":
        vae = AutoencoderKL.from_single_file(main_dir + "VAE/" + vae_name, torch_dtype=torch_dtype_).to(device_) # torch.float16 gives random black image
        pipeline.vae = vae
        vae_hash=calculate_sha256(main_dir + "VAE/" + vae_name, 10)

    # LoRA
    if sd_xl:
        model_path_lora = main_dir + "Lora/"
    elif sd1_5_dir:
        model_path_lora = main_dir + "Lora/1.5/"   
    else:
        model_path_lora = main_dir + "Lora/"

    if turbo_lcm:
        if sd_xl:
            model_file_lora = ["sd_xl_turbo_lora_v1-128dim.safetensors"]
        else:
            model_file_lora = ["LCM_LoRA_Weights_SD15.safetensors"]
    else:
        model_file_lora = os.getenv("model_file_lora", "").split(",")

    lora_w = os.getenv("lora_w", "1").split(",")

    if model_file_lora[0] != "":  
        i = 0
        adapters = []
        adapter_weights = []
        for model_file_lora_single in model_file_lora:
            i = i + 1
            pipeline.load_lora_weights(model_path_lora, weight_name=model_file_lora_single, adapter_name=str(i))
            adapters.append(str(i))
            adapter_weights.append(float(lora_w[i-1]))
            with open(model_path_lora + model_file_lora_single, "rb") as file:
                if i == 1:
                    txt_file_lora = ', Lora hashes: "' + os.path.splitext(os.path.basename(model_file_lora_single))[0] + ': ' + addnet_hash_safetensors(file, 12)       
                else:
                    txt_file_lora = txt_file_lora + ", " + os.path.splitext(os.path.basename(model_file_lora_single))[0] + ': ' + addnet_hash_safetensors(file, 12)
        txt_file_lora = txt_file_lora + '"'  
        pipeline.set_adapters(adapters, adapter_weights=adapter_weights)
        # Fuses the LoRAs into the Unet
        pipeline.fuse_lora()
    else:
        txt_file_lora = ""   

    # End Lora
        
    if not sd_xl:
        if clip_skip > 0:
            clip_layers = pipeline.text_encoder.text_model.encoder.layers
            pipeline.text_encoder.text_model.encoder.layers = clip_layers[:-clip_skip]

    lcm = os.getenv("lcm", "false").lower() == "true"
    if turbo_lcm or lcm:
        eta = 0
        pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)
        Sampler = "LCM"
    else:
        eta = 0 
        # if pipeline.scheduler == DDIMScheduler: eta = 0.31337
        euler_a = os.getenv("euler_a", "false").lower() == "true"
        if euler_a:
            pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
            Sampler = "Euler a"
        else:
            dpm_multi = os.getenv("dpm_multi", "false").lower() == "true"
            if dpm_multi:
                pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, use_karras_sigmas='true')
                Sampler = "DPM++ 2M Karras"
            else:
                pipeline.scheduler = DPMSolverSinglestepScheduler.from_config(pipeline.scheduler.config, use_karras_sigmas='true')
                Sampler = "DPM++ SDE Karras"

    # pipeline.unet = torch.compile(pipeline.unet, mode="reduce-overhead", fullgraph=True) # RuntimeError: Windows not yet supported for torch.compile

    embeddings = os.getenv("embeddings", "").split(",")
    if embeddings[0] == "": 
        embeddings = os.getenv("embedding", "").split(",")
        
    i = 0
    if embeddings[0] != "":  
        for embedding in embeddings:
            i = i + 1
            pipeline.load_textual_inversion("./embeddings/"+embedding, token=os.path.splitext(os.path.basename(embedding))[0])
            if i == 1:
                txt_file_embedding = ', TI hashes: "' + os.path.splitext(os.path.basename(embedding))[0] + ': ' + calculate_sha256("./embeddings/"+embedding, 12)      
            else:
                txt_file_embedding = txt_file_embedding + ", " + os.path.splitext(os.path.basename(embedding))[0] + ': ' + calculate_sha256("./embeddings/"+embedding, 12) 
        txt_file_embedding = txt_file_embedding + '"'  
    else:
        txt_file_embedding = ""   

    pipeline.safety_checker = None
    pipeline.requires_safety_checker = False
    pipeline.to(device_)

    max_n_images_gif = 50
    if cycles < max_n_images_gif:
        max_n_images_gif = cycles
    x_cycle = 1

    if best_prompt_search:
        cycles = 1 # end cycles after guidance_scale = 7 or 11
    

    while cycles == 0 or x_cycle <= cycles or best_prompt_search:
        if cycles > 0:
            print(f"Image n.{x_cycle} of {cycles}")
  
        if turbo_lcm:
            if sd_xl:
                width_ = 768 # 512
                height_ = 1024 # 768
            else: #1.5 lcm
                width_ = 512
                height_ = 768
        elif random_dimension:
            if sd_xl:          
                width_ = random.choices([768, 832, 1216, 1280], weights=[0.4, 0.4, 0.1, 0.1], k=1)[0]
                match width_:
                    case 768:
                        height_ = 1280
                    case 832:
                        height_ = 1216
                    case 1216:
                        height_ = 832
                    case 1280:
                        height_ = 768
            else: #1.5
                width_ = random.choices([512, 768, 1024], weights=[0.8, 0.1, 0.1], k=1)[0]
                match width_:
                    case 512:
                        height_ = 768
                    case 768:
                        height_ = random.choice([512,1024])
                    case 1024:
                        height_ = 768              
        elif sd_xl:
            width_ = int(os.getenv("width_xl", "832"))
            height_ = int(os.getenv("height_xl", "1216"))
        else: #1.5
            width_ = int(os.getenv("width", "512"))
            height_ = int(os.getenv("height", "768"))


        if best_prompt_search:
            if turbo_lcm:
                guidance_scale = 1 + ((x_cycle - 1) * 0.5) # from 1 to 7
                if guidance_scale == 7:
                    best_prompt_search = False
            else:
                guidance_scale = 5 + ((x_cycle - 1) * 0.5) # from 5 to 11
                if guidance_scale == 11:
                    best_prompt_search = False
        elif turbo_lcm:  
            if sd_xl:
                guidance_scale = 1.5 # default = 2, sd_xl turbo with guidance scale = from 0 to 3.5
            else:
                guidance_scale = 2.5 # default = 2.5, sd 1.5 lcm with guidance scale ~2.5
        else:
            guidance_scale = float(os.getenv("guidance_scale", "7"))
            if guidance_scale.is_integer():
                guidance_scale = int(guidance_scale)

        if sd_xl:
            if turbo_lcm:
                num_inference_steps_= 8 # 4 or 8 with turbo sd_xl lora, 8 or 16 with lcm sd_xl
            else:
                num_inference_steps_= int(os.getenv("steps_xl", "80"))
        else: # 1.5
            if turbo_lcm:
                num_inference_steps_= 16 # 8 or 16 with lcm 1.5 lora
            else:
                num_inference_steps_= int(os.getenv("steps", "40"))
        if sd_xl:
            num_images_per_prompt_ = int(os.getenv("num_images_per_prompt_xl","1"))
        else:
            num_images_per_prompt_ = int(os.getenv("num_images_per_prompt","1"))

        prompt = ""
        if (prompt_action) and (prompt_input != ""):
            prompt_action_input = prompt_input
            while re.search(r'__', prompt_action_input):
                re_sep = r"(__.*?__)"
                
                substrings = re.split(re_sep, prompt_action_input, maxsplit=1)

                if len(substrings) > 1:
                    wildcard_word = substrings[1].replace("__","")
                    prompt = prompt + substrings[0] + random_wild(wildcard_word,"",True)
                    prompt_action_input = substrings[2]
                else:
                    break
            prompt = prompt + prompt_action_input
        else:
            prompt = prompt_input

        if dynamic_prompt > 0:
            if prompt[-1] != ",":
                prompt = prompt + ","
            banned_words = os.getenv("banned_words", "").split(",")
            prompt = app_retnet.main_def(prompt_input=prompt, max_tokens=dynamic_prompt, DEVICE="cpu", banned_words=banned_words)
            prompt = remove_duplicates(prompt)

        prompt = remove_last_comma(prompt)
        prompt = prompt.strip()
        negative_prompt = negative_prompt.strip()

        compel_enabled = (os.getenv("compel","true").lower() == "true")
        if compel_enabled:
            if sd_xl:
                from compel import Compel, ReturnedEmbeddingsType
                compel = Compel(
                tokenizer=[pipeline.tokenizer, pipeline.tokenizer_2] ,
                text_encoder=[pipeline.text_encoder, pipeline.text_encoder_2],
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=[False, True],
                truncate_long_prompts=truncation_option
                )
                conditioning, pooled = compel.build_conditioning_tensor(prompt)
                negative_conditioning, negative_pooled = compel.build_conditioning_tensor(negative_prompt)
                [conditioning, negative_conditioning] = compel.pad_conditioning_tensors_to_same_length([conditioning, negative_conditioning])
            else:
                from compel import Compel
                compel = Compel(tokenizer=pipeline.tokenizer, text_encoder=pipeline.text_encoder,truncate_long_prompts=truncation_option)
                conditioning = compel.build_conditioning_tensor(prompt)
                negative_conditioning = compel.build_conditioning_tensor(negative_prompt)
                [conditioning, negative_conditioning] = compel.pad_conditioning_tensors_to_same_length([conditioning, negative_conditioning])
        # else:
            # if not sd_xl:
            #    conditioning, negative_conditioning = get_pipeline_embeds(pipeline, prompt, negative_prompt, device_, truncation_option)

        if random_seed and best_prompt_search == False: 
            input_seed = random.randint(0, 9999999999)
        else:
            input_seed = int(input_seed)
        
        generator_ = torch.Generator(device=device_).manual_seed(input_seed)

        print("Prompt: " + prompt)
        
        resize_pixel_w = width_ % 8
        resize_pixel_h = height_ % 8
        if resize_pixel_w > 0:
            width_ = width_ - resize_pixel_w
        if resize_pixel_h > 0:
            height_ = height_ - resize_pixel_h
        while True:
            start_time = time.time()
            if compel_enabled:
                if sd_xl:
                    images = pipeline(prompt_embeds=conditioning, negative_prompt_embeds=negative_conditioning, pooled_prompt_embeds=pooled, negative_pooled_prompt_embeds=negative_pooled, generator=generator_, eta=eta, width=width_, height=height_, num_inference_steps=num_inference_steps_, guidance_scale=guidance_scale,
                        num_images_per_prompt=num_images_per_prompt_).images
                else:
                    images = pipeline(prompt_embeds=conditioning, negative_prompt_embeds=negative_conditioning, generator=generator_, eta=eta, width=width_, height=height_, num_inference_steps=num_inference_steps_, guidance_scale=guidance_scale,
                        num_images_per_prompt=num_images_per_prompt_).images
            else:
                # if sd_xl:
                    images = pipeline(prompt=prompt, negative_prompt=negative_prompt, generator=generator_, eta=eta, width=width_, height=height_, num_inference_steps=num_inference_steps_, guidance_scale=guidance_scale,
                        num_images_per_prompt=num_images_per_prompt_).images
                # else:
                #    images = pipeline(prompt_embeds=conditioning, negative_prompt_embeds=negative_conditioning, generator=generator_, eta=eta, width=width_, height=height_, num_inference_steps=num_inference_steps_, guidance_scale=guidance_scale,
                #       num_images_per_prompt=num_images_per_prompt_).images



            end_time = time.time()

            duration = end_time - start_time

            print(f"Time: {duration} seconds.")

            if are_all_not_black(images) or pipeline.safety_checker != None: # with safety checker enabled higher probabily of black images
                break

        if resize_pixel_w > 0:
            width_ = width_ + resize_pixel_w
        if resize_pixel_h > 0:
            height_ = height_ + resize_pixel_h

        if no_vae or vae_name == "":
           vae_string = ""
        else:
            vae_string = ", VAE hash: " + vae_hash + ", VAE: "+ vae_name # vae_name with extension at the end!

        txt_file_data = ""
        if model_file_lora[0] != "":
            i = 0
            for model_file_lora_single in model_file_lora:
                if float(lora_w[i]).is_integer():
                    lora_w_number = int(lora_w[i])
                else:
                    lora_w_number = str(lora_w[i])
                txt_file_data = txt_file_data + f"<lora:{os.path.splitext(os.path.basename(model_file_lora_single))[0]}:{lora_w_number}>"
                i = i + 1
        else:
            txt_file_data = ""
        txt_file_data=txt_file_data+prompt+"\n"+"Negative prompt: "+negative_prompt+"\n"+"Steps: "+str(num_inference_steps_)+", Sampler: "+Sampler+", CFG scale: "+str(guidance_scale)+", Seed: "+str(input_seed)+", Size: "+str(width_)+"x"+str(height_)+", Model hash: "+model_hash+", Model: "+os.path.splitext(os.path.basename(checkpoint))[0]+vae_string+", Clip skip: "+str(clip_skip)

        if txt_file_lora != "":
            txt_file_data = txt_file_data+txt_file_lora

        if txt_file_embedding != "":
            txt_file_data = txt_file_data+txt_file_embedding

        if eta > 0:
            txt_file_data = txt_file_data+", Eta: " + str(eta)

        print(txt_file_data)
        
        if env_loaded:
            if env_file == "file":
               with open("./env/merged_file.env", 'r') as env_file_data:
                    env_file_text = env_file_data.read()
            else:
                with open(env_file, 'r') as env_file_data:
                    env_file_text = env_file_data.read()
        else:
            env_file_text = ""

        FirstImage = True
        for image in images:
            if resize_pixel_w > 0 or resize_pixel_h > 0:
                image = image.resize((width_, height_))
            if (x_cycle == 1) and (FirstImage):
                FirstImage = False
                if create_story and device_ == "cuda":
                    torch.cuda.empty_cache()
                if directory_save == "":
                    file_path = image_save_file.save_file(image, txt_file_data, True, create_story, prompt, env_file, env_file_text)
                else:
                    file_path = image_save_file.save_file_alt(image, txt_file_data, directory_save,env_file_text)
                if create_story and device_ == "cuda":
                    torch.cuda.empty_cache()
            else:
                if directory_save == "":
                    file_path = image_save_file.save_file(image, txt_file_data, False, False, "", env_file, env_file_text)
                else:
                    file_path = image_save_file.save_file_alt(image, txt_file_data, directory_save, env_file_text)
                FirstImage = False
        if gif_creation:
            if cycles > 1 and x_cycle == max_n_images_gif and file_path != "": # at least 2 images, max 50 images
                create_gif.create_gif_def(os.path.dirname(file_path), os.path.splitext(os.path.basename(file_path))[0], max_n_images_gif)
        x_cycle = x_cycle + 1

    if model_file_lora[0] != "": 
        pipeline.unfuse_lora()  

    # del pipeline
    # gc.collect()
    # torch.cuda.empty_cache()

    return image

def main_def(env_loaded = False, env_file="./env/.env", main_dir="./", prompt_input="", cycles=0, dynamic_prompt=-1, directory_save=""):
    ################## main option ##################
    if not env_loaded:
        if os.path.isfile("./env/merged_file.env"):
            os.remove("./env/merged_file.env")
        if env_file == "./env/.env":
            try:
                load_dotenv(env_file) 
            except:
                env_loaded = False
                print(f"Error! {env_file} not loaded")
            else:
                env_loaded = True
                print(f"{env_file} loaded")
        else:
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

    if main_dir == "" or main_dir == "./":
        main_dir=os.getenv("main_dir","./")
        if main_dir[-1] != "/":
            main_dir = main_dir + "/"

        if prompt_input=="":
            prompt_input = os.getenv("prompt_input","1girl")

        if cycles==0:
            cycles = int(os.getenv("cycles", "10"))
        if dynamic_prompt==-1:
            dynamic_prompt = int(os.getenv("dynamic_prompt", "0")) # number means max new token, 64 = default, 0 = off

    negative_prompt = os.getenv("negative_prompt", "")
    prompt_action = os.getenv("prompt_action", "true").lower() == "true"
    sd_xl = os.getenv("sd_xl", "true").lower() == "true"
    clip_skip = int(os.getenv("clip_skip", "1"))
    turbo_lcm = os.getenv("turbo_lcm", "false").lower() == "true"
    no_vae = os.getenv("no_vae", "false").lower() == "true"
    random_seed = os.getenv("random_seed", "true").lower() == "true"
    input_seed = int(os.getenv("input_seed", "1"))
    truncation_option = os.getenv("truncation_option", "false").lower() == "true"
    gif_creation = os.getenv("gif_creation", "false").lower() == "true"
    create_story = os.getenv("create_story", "false").lower() == "true"
    random_dimension = os.getenv("random_dimension","false").lower() == "true"
    best_prompt_search = os.getenv("best_prompt_search", "false").lower() == "true" # Same seed, same prompt, same negative prompt; guidance_scale from 1-5 to 7-11
    if best_prompt_search:
        print("best_prompt_search ENABLED")

    return image_print(
            env_loaded = env_loaded,
            env_file = env_file,
            main_dir=main_dir,
            prompt_action=prompt_action,
            prompt_input=prompt_input,
            negative_prompt=negative_prompt,
            sd_xl=sd_xl,
            clip_skip=clip_skip,
            turbo_lcm=turbo_lcm,
            no_vae=no_vae,
            dynamic_prompt=dynamic_prompt,
            random_seed=random_seed,
            input_seed=input_seed,
            cycles=cycles,
            truncation_option=truncation_option,
            gif_creation=gif_creation,
            create_story=create_story,
            best_prompt_search=best_prompt_search,
            random_dimension=random_dimension,
            directory_save=directory_save)

if __name__ == "__main__":
    ############ advanced option ############
    """
    if (len(sys.argv) == 3) and ((sys.argv[1] == "--audio") or (sys.argv[1] == "-audio") or (sys.argv[1] == "--a") or (sys.argv[1] == "-a")):
        descriptions_ = [sys.argv[2]]
        app_audio.main_def(descriptions_)
    else:
    """
    if os.path.isfile("./env/merged_file.env"):
        os.remove("./env/merged_file.env")
    try:
        if len(sys.argv) > 2:
            env_file_base = f"./env/{sys.argv[1]}"
            if env_file_base[-4:] != ".env" and env_file_base[-5:] != ".yaml":
                env_file_base = f"{env_file_base}.env"
            if not os.path.isfile(env_file_base):
                env_file_base = "./env/.env"

            env_file = f"./env/{sys.argv[2]}"
            if env_file[-4:] != ".env" and env_file[-5:] != ".yaml":
                env_file = f"{env_file}.env"
            if not os.path.isfile(env_file):
                env_file = "./env/.env"

            with open(env_file_base, 'r') as file1, open(env_file, 'r') as file2:
                content1 = file1.read()
                content2 = file2.read()

            merged_content = content1 + "\n" + content2

            with open("./env/merged_file.env", 'w') as merged_file:
                merged_file.write(merged_content)
                import subprocess
                subprocess.run(["attrib","+H","./env/merged_file.env"],check=True)

            load_dotenv("./env/merged_file.env") # env_file overwrite env_file_base
            env_loaded = True
            print("merged_file.env loaded")
        elif len(sys.argv) == 2:
            if sys.argv[1] == "file":
                # Carica l'immagine PNG
                image_path = input("Insert image path: ")
                image = Image.open(image_path)

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
            else:
                env_file = f"./env/{sys.argv[1]}"
                if env_file[-4:] != ".env" and env_file[-5:] != ".yaml":
                    env_file = f"{env_file}.env"
                if not os.path.isfile(env_file):
                    env_file = "./env/.env"

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
    cycles = int(os.getenv("cycles", "10"))
    dynamic_prompt = int(os.getenv("dynamic_prompt", "0")) # number means max new token, 64 = default, 0 = off
    directory_save = ""
    main_dir=os.getenv("main_dir","./")
    if main_dir[-1] != "/":
        main_dir = main_dir + "/"
    main_def(env_loaded=env_loaded,env_file=env_file,main_dir=main_dir,prompt_input=prompt_input,cycles=cycles,dynamic_prompt=dynamic_prompt,directory_save=directory_save)
    if os.path.isfile("./env/merged_file.env"):
        os.remove("./env/merged_file.env")
