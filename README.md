# Based on Fined-tuned <a herf="https://github.com/tatsu-lab/stanford_alpaca" target="_blank">Stanford Alpaca </a>, further tune it on <a href="https://www.consumptionvoucher.gov.hk/public/pdf/2023cvs/FAQ-2023_en.pdf" target="_blank">Hong Kong 2023 Consumption Voucher Scheme Frequently Asked Questions</a> Dataset

This repo works on the top of https://github.com/tloen/alpaca-lora reproducing the [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) results using [low-rank adaptation (LoRA)](https://arxiv.org/pdf/2106.09685.pdf).
We provide an Instruct model of similar quality to `text-davinci-003` that can run [on a Raspberry Pi](https://twitter.com/miolini/status/1634982361757790209) (for research),
and the code can be easily extended to the `13b`, `30b`, and `65b` models.

### Here's [a colab notebook](https://colab.research.google.com/drive/156_XkkJc4-jFXebBg1VqETIi8VHhjFUY?usp=sharing) to see the difference between before and after fine tune on the HK CVS FQA datasets

### Setup

```
pip install -r requirements.txt
```

### Prepare your Dataset (`prepare_your_dataset.ipynb`)

The downstream task is to enable the model to answer related <a href="https://www.consumptionvoucher.gov.hk/public/pdf/2023cvs/FAQ-2023_en.pdf" target="_blank">Hong Kong 2023 Consumption Voucher Scheme Frequently Asked Questions</a>. You may change the source of dataset that fit your personal need.

### Fine Tuning (`finetune.ipynb`)

We fined tune the model based on the fine-tuned version of [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) and reproduced by this repo: [alpaca-lora](https://github.com/tloen/alpaca-lora) using Hugging Face's [PEFT](https://github.com/huggingface/peft) to fine tune it cheaply and efficiently.

### Inference

We push our prompt tuning adaptor into huggingface hub and use it wrap the [decapoda-research/llama-7b-hf](https://huggingface.co/decapoda-research/llama-7b-hf)

Run the code

```python

from transformers import LlamaForCausalLM, LlamaTokenizer,GenerationConfig
from peft import PeftModel


device_map = "auto"

tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
model = LlamaForCausalLM.from_pretrained(
    "decapoda-research/llama-7b-hf",
    load_in_8bit=True,
    device_map="auto",
)

### Load model after fine tuned on our datasets
model = PeftModel.from_pretrained(model, "Nelsonlin0321/alpaca-lora-7b-tuned-on-hk-cvs-fqa")

tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
tokenizer.pad_token_id = 0


def generate_prompt_eval(instruction):
    template =  f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
{instruction}
### Response:"""
    return template

eval_generation_config = GenerationConfig(
    temperature=0.1,
    top_p=0.75,
    num_beams=4,
)


def generate_answer(instruction):
    prompt = generate_prompt_eval(instruction)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()
    generation_output = model.generate(
        input_ids=input_ids,
        generation_config=eval_generation_config,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=256
    )
    for s in generation_output.sequences:
        output = tokenizer.decode(s)
        print("Response:", output.split("### Response:")[1].strip())


question = "Who are eligible to be disbursed with the first-instalment voucher of $1,500 on 16 April?"

generate_answer(question)
>> Response: All eligible people who have successfully registered under 2022 CVS and met the relevant eligibility criteria will be disbursed with the first-instalment voucher of $1,500 on 16 April.

```
