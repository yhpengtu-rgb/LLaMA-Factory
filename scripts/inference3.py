
import gc, re
import json, os, sys
from typing import Optional
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src/llamafactory'))
sys.path.append(project_root)

import fire
import numpy as np
from tqdm import tqdm
from transformers import Seq2SeqTrainingArguments

from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.extras.misc import get_device_count
from llamafactory.extras.packages import is_vllm_available
from llamafactory.hparams import get_infer_args
from llamafactory.model import load_tokenizer
from llamafactory.data.loader import get_dataset_and_preprocessor

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

if __name__ == "__main__":
    model_name_or_path = '/inspire/ssd/project/sais-bio/public/tupeng/LLaMA-Factory/huggingface/weights/LLM-Research/Meta-Llama-3-8B-Instruct'
    adapter_name_or_path = '/inspire/ssd/project/sais-bio/public/tupeng/LLaMA-Factory/saves/_llama3-8b/qlora/sft/checkpoint-30000'

    dataset = 'alpaca_valid_proteins'
    template = 'llama3'
    cutoff_len = 9000
    dataset_dir = 'data'
    max_samples = 1000
    default_system = None
    enable_thinking = True
    temperature: float = 1.5
    top_p: float = 0.95
    top_k: int = 100
    max_new_tokens: int = 1024
    repetition_penalty: float = 1.0
    pipeline_parallel_size: int = 1
    skip_special_tokens: bool = True
    seed: Optional[int] = None
    batch_size: int = 1

    model_args, data_args, _, generating_args = get_infer_args(
        dict(
            model_name_or_path=model_name_or_path,
            adapter_name_or_path=adapter_name_or_path,
            dataset=dataset,
            dataset_dir=dataset_dir,
            template=template,
            cutoff_len=cutoff_len,
            max_samples=max_samples,
            preprocessing_num_workers=16,
            default_system=default_system,
            enable_thinking=enable_thinking,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
        )
    )

    training_args = Seq2SeqTrainingArguments(output_dir="dummy_dir")
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template_obj = get_template_and_fix_tokenizer(tokenizer, data_args)
    template_obj.mm_plugin.expand_mm_tokens = False  # for vllm generate

    engine_args = {
        "model": model_args.model_name_or_path,
        "trust_remote_code": True,
        "dtype": model_args.infer_dtype,
        # "max_model_len": cutoff_len + max_new_tokens,
        "tensor_parallel_size": (get_device_count() // pipeline_parallel_size) or 1,
        "pipeline_parallel_size": pipeline_parallel_size,
        "disable_log_stats": True,
        "enable_lora": model_args.adapter_name_or_path is not None,
    }

    llm = LLM(**engine_args)

    # load datasets
    dataset_module, dataset_processor = get_dataset_and_preprocessor(template_obj, model_args, data_args, training_args, "ppo", **tokenizer_module)
    train_dataset = dataset_module["train_dataset"]

    sampling_params = SamplingParams(
        repetition_penalty=generating_args.repetition_penalty or 1.0,  # repetition_penalty must > 0
        temperature=generating_args.temperature,
        top_p=generating_args.top_p or 1.0,  # top_p must > 0
        top_k=generating_args.top_k or -1,  # top_k must > 0
        stop_token_ids=template_obj.get_stop_token_ids(tokenizer),
        max_tokens=generating_args.max_new_tokens,
        skip_special_tokens=skip_special_tokens,
        seed=seed,
    )
    if model_args.adapter_name_or_path is not None:
        lora_request = LoRARequest("default", 1, model_args.adapter_name_or_path[0])
    else:
        lora_request = None

    # Store all results in these lists
    n_steps = 10000
     # Add batch process to avoid the issue of too many files opened
    data_bar = tqdm(range(0, len(train_dataset), batch_size), total=len(train_dataset))
    for i in data_bar:
        all_prompts, all_preds, all_labels, all_trajectories = [], [], [], []
        # if i != 0 or \
        #     i != len(train_dataset)-1 or i != len(train_dataset)-2 or i != len(train_dataset)-3 or i != len(train_dataset)-4: continue
        # print(i)
        # a = len(train_dataset)
        data_bar.set_description(f'Processing batched inference, {i+1}/{len(train_dataset)} proteins ...')
        if i==0 or i==1 or i==2 or i==3 or i == len(train_dataset)-2 or i == len(train_dataset)-3 or i == len(train_dataset)-4 or i == len(train_dataset)-1:
            # if i != len(train_dataset)-1: continue
            
            save_folder = f'./results_{temperature}_{top_p}_{repetition_penalty}/100Proteins_10ps'
            if not os.path.exists(save_folder): os.makedirs(save_folder, exist_ok=True)
            save_name = os.path.join(save_folder, f'{i}-th_protein_trajectory.npy')
            if os.path.exists(save_name): continue
            
            vllm_inputs, prompts, labels, all_trajectories = [], [], [], []
            batch = train_dataset[i : min(i + batch_size, len(train_dataset))]
            
            for step in range(n_steps):
                print(f'Protein idxes: {i}, {step+1}/{n_steps}-th trajectories ...')
                if step == 0:
                    for j in range(len(batch["input_ids"])):
                        multi_modal_data = None

                        vllm_inputs.append({"prompt_token_ids": batch["input_ids"][j], "multi_modal_data": multi_modal_data})
                        prompts.append(tokenizer.decode(batch["input_ids"][j], skip_special_tokens=skip_special_tokens))
                        labels.append(
                            tokenizer.decode(
                                list(filter(lambda x: x != IGNORE_INDEX, batch["labels"][j])),
                                skip_special_tokens=skip_special_tokens,
                            )
                        )
                        
                    
                else:
                    vllm_inputs.append({"prompt_token_ids": batch["input_ids"][0], "multi_modal_data": multi_modal_data})
                    prompts.append(tokenizer.decode(batch["input_ids"][0], skip_special_tokens=skip_special_tokens))
                        
                results = llm.generate([vllm_inputs[-1],], sampling_params, lora_request=lora_request)
                preds = [result.outputs[0].text for result in results]
                
                text = prompts[-1].split('\n')
                last_frame = re.search(r"Frame 9:\s*\[(.*?)\]", text[12])
                last_numbers = re.findall(r"\d+", last_frame.group(1)) #re.findall(r'\d+', frame_str)
                last_numbers = [int(n) for n in last_numbers if n.isdigit()]
                
                pred_text = preds[-1]
                pred_frame = re.search(r"Frame 10:\s*\[(.*?)\]", pred_text)
                pred_numbers = re.findall(r"\d+", pred_frame.group(1)) #re.findall(r'\d+', frame_str)
                pred_numbers = [int(n) for n in pred_numbers if n.isdigit()]
                
                while len(pred_numbers) != len(last_numbers):
                    results = llm.generate(vllm_inputs, sampling_params, lora_request=lora_request)
                    preds = [result.outputs[0].text for result in results]
                    
                    text = prompts[-1].split('\n')
                    last_frame = re.search(r"Frame 9:\s*\[(.*?)\]", text[12])
                    last_numbers = re.findall(r"\d+", last_frame.group(1)) #re.findall(r'\d+', frame_str)
                    last_numbers = [int(n) for n in last_numbers if n.isdigit()]
                    
                    pred_text = preds[-1]
                    pred_frame = re.search(r"Frame 10:\s*\[(.*?)\]", pred_text)
                    pred_numbers = re.findall(r"\d+", pred_frame.group(1)) #re.findall(r'\d+', frame_str)
                    pred_numbers = [int(n) for n in pred_numbers if n.isdigit()]
                
                if step == 0:
                    for k, frame in enumerate(text[4:13], 1):
                        match = re.search(r":\s*\[(.*?)\]", frame)
                        numbers = re.findall(r"\d+", match.group(1))
                        all_trajectories.append([int(n) for n in numbers if n.isdigit()])
                        
                    all_trajectories.append(pred_numbers)
                else:
                    all_trajectories.append(pred_numbers)
                    
                frames = text[5:13]
                frames.append(preds[-1])
                
                renumbered_frames = []
                for k, frame in enumerate(frames, 1):
                    numbers = re.findall(r'\[.*\]', frame)[0]
                    renumbered_frames.append(f"Frame {k}: {numbers};")
                
                # renumbered_frames[-1] = renumbered_frames[-1]+'assistant'
                text[4:13] = renumbered_frames
                new_text = '\n'.join(text[2:13])
                
                # print(prompts[-1])
                # print(new_text)
                
                response_tmp = re.findall(r'\[.*\]', frames[-1])[0]
                response_tmp = f"Frame 10: {response_tmp};"
                
                example_prompt = [dict(content=new_text, role='user')]
                example_response = [dict(content=response_tmp, role='assistant')]
                messages = example_prompt + example_response
                messages = dataset_processor.template.mm_plugin.process_messages(messages, [], [], [], dataset_processor.processor)
                new_input_ids, _ = dataset_processor.template.encode_oneturn(dataset_processor.tokenizer, messages)
                assert len(new_input_ids) == len(batch['input_ids'][0])
                
                batch['input_ids'][0] = new_input_ids                
            
            a = np.asarray(all_trajectories)
            # save_folder = f'./results_{temperature}_{top_p}_{repetition_penalty}/100Proteins_10ps'
            # if not os.path.exists(save_folder): os.makedirs(save_folder, exist_ok=True)
            # save_name = os.path.join(save_folder, f'{i}-th_protein_trajectory.npy')
            # if os.path.exists(save_name): 
            #     os.remove(save_name)
            
            np.save(save_name, np.asarray(all_trajectories))
