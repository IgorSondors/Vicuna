import time
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastchat.model import get_conversation_template
from fastchat.conversation import *

def load_model(model_name, device, num_gpus, load_8bit=False):
    if device == "cuda":
        kwargs = {"torch_dtype": torch.float16}
        if load_8bit:
            if num_gpus != "auto" and int(num_gpus) != 1:
                print("8-bit weights are not supported on multiple GPUs. Revert to use one GPU.")
            kwargs.update({"load_in_8bit": True, "device_map": "auto"})
        else:
            if num_gpus == "auto":
                kwargs["device_map"] = "auto"
            else:
                num_gpus = int(num_gpus)
                if num_gpus != 1:
                    kwargs.update({
                        "device_map": "auto",
                        "max_memory": {i: "19GiB" for i in range(num_gpus)},
                    })
    elif device == "cpu":
        kwargs = {}
    else:
        raise ValueError(f"Invalid device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast = False)
    model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True,
        low_cpu_mem_usage=True, **kwargs)

    # calling model.cuda() mess up weights if loading 8-bit weights
    if device == "cuda" and num_gpus == 1 and not load_8bit:
        model.cuda()

    return model, tokenizer

def prompt(model, tokenizer, text, temperature, context):
    #conv = get_conversation_template(context)
    conv = get_conv_template(context)
    conv.append_message(conv.roles[0], text)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer([prompt]).input_ids
    output_ids = model.generate(
        torch.as_tensor(input_ids).cuda(),
        do_sample=True,
        temperature=temperature,#0.7
        max_new_tokens=40#1024,
    )

    output_ids = output_ids[0][len(input_ids[0]) :]
    return tokenizer.decode(output_ids, skip_special_tokens=True).strip()

def iterate_over_df(df, model, tokenizer, temperature, context):
    prompts_list = [0 for i in range(len(df['name']))]
    cnt = 0
    start_time = time.time()
    for index, row in df.iterrows():
        
        name = row['name']
        #text = f"Imagine that you can only parse named entities from text and nothing else. Parse the information from the following tablet description '{name}' and based on that complete gaps from next statements: \nBrand name:___\nDevice:___\nColor:___.\nYour answer should have exactly the template I provided you and no more other words. Please double check your answer before you give me that."
        answer = prompt(model, tokenizer, name, temperature, context)

        #prompts_list.append(answer)
        prompts_list[cnt] = answer
        cnt += 1
        print(cnt)
        if cnt < 10:
            print(answer)
        
    print('time_spent = ', time.time() - start_time) 
    print('mean_time = ', (time.time() - start_time)/cnt)    
    df['prompt'] = prompts_list
    return df

def limit_balace_offers(df):
    df = df.drop_duplicates(subset=['name'], keep="first", inplace=False)
    ans = [y for x, y in df.groupby('model_id')]
    for i in range(len(ans)):
        ans[i] = ans[i].sample(frac=1).sample(frac=1)[:9]
    df = pd.concat(ans, axis=0)
    return df

if __name__ == "__main__":
    model_name = "/mnt/vdb1/ggml_vicuna_13b_8bit"
    device = "cuda"
    num_gpus=1
    load_8bit = False
    model, tokenizer = load_model(model_name, device, num_gpus, load_8bit)
    csv_name = '/mnt/vdb1/offer-dt-230417-165415.csv'
    df = pd.read_csv(csv_name, sep=';')
    #df_planshet = df[df.category_name == 'планшетные компьютеры и мини-планшеты']
    #df_planshet = df[df.category_name == 'мобильные телефоны']
    df_planshet = df[:10000]   # Делаем первые 30к
    #df_planshet = limit_balace_offers(df_planshet)
    #df_planshet = df_planshet[:30000]   # Делаем первые 30к
    temperature = 0.01
    
    context = "obuv"
    print(f"temperature = {temperature}, context = {context}")
    df_planshet_new = iterate_over_df(df_planshet, model, tokenizer, temperature, context)
    df_planshet_new.to_csv('/mnt/vdb1/obuv.csv', sep=';',index=False)
    
    # context = "planshet_small"
    # print(f"temperature = {temperature}, context = {context}")
    # df_planshet_new = iterate_over_df(df_planshet, model, tokenizer, temperature, context)
    # df_planshet_new.to_csv('/mnt/vdb1/planshet_prompts_0temp_context_planshet_small.csv', sep=';',index=False)

    # context = "planshet_big"
    # print(f"temperature = {temperature}, context = {context}")
    # df_planshet_new = iterate_over_df(df_planshet, model, tokenizer, temperature, context)
    # df_planshet_new.to_csv('/mnt/vdb1/planshet_prompts_0temp_context_planshet_big.csv', sep=';',index=False)

    
