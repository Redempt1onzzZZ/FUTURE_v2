from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name_or_path = './CodeLlama-13b-Python-hf' # change this to your model path

adapter_name_or_path = './results/final_checkpoint' # change this to your finetuned results path
save_path = './combined_model' # change this to your save path

tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
    trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    device_map='auto'
)
print("load model success")
model = PeftModel.from_pretrained(model, adapter_name_or_path)
print("load adapter success")
model = model.merge_and_unload()
print("merge success")

tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)
print("save done.")
