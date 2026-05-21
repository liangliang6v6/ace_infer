from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import argparse
import os 
from tqdm import trange
import copy
import json
import re

parser = argparse.ArgumentParser("")
parser.add_argument("--tokenizer", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
parser.add_argument("--model_path", type=str)
parser.add_argument("--expname", type=str, default="")
parser.add_argument("--temperature", type=float, default=0.0)
parser.add_argument("--top_p", type=float, default=0.99)
parser.add_argument("--tensor_parallel_size", type=int, default=1) # number of GPU
parser.add_argument("--datasets", type=str, default="hotpotqa") # number of GPU

args = parser.parse_args()

qwen_prompt_plan = """
You are an expert at breaking down complex questions and claims into a logical sequence of simpler, atomic sub-questions. There are noisy constrains in the question, please identify the most selective relations among them and use those as the subquestions.
Please break down the following input into multiple specific sub-questions that address individual components of the original statement, order the subquestions for potential easier retrieval.

### Rules:
1. You MUST mark every single sub-question with "### Q<number>:" at the beginning. 
2. If you need to refer to answers from earlier sub-questions, use #1, #2, etc., to indicate the corresponding answers.
3. Do not include any introductory or concluding text. Output ONLY the decomposed questions.

### Example:
Input: Who is the maternal grandfather of the individual whose mother held the title of Duchess, and whose brother, Claud Andrew Montagu Douglas Scott, was educated at Eton College, received the Distinguished Service Order, and was also an Officer of the Order of Saint John?
Decomposed Question:
### Q1: Who is Lord Herbert Montagu Douglas Scott's mother?
### Q2: Who is the father of #1?

### Your Task:
Input: {question}
Decomposed Question:
"""

'''original prompt'''
prompt_plan_qa = """Please break down the question "{question}" into multiple specific sub-questions that address individual components of the original question. 
Mark each sub-question with ### at the beginning.  If you need to refer to answers from earlier sub-questions, use #1, #2, etc., to indicate the corresponding answers.
Decomposed Question:"""

if args.expname == "qwen":
    print("Using Qwen prompt plan")
    prompt_plan_qa = qwen_prompt_plan


prompt_plan_claim = """Please break down the claim "{claim}" into multiple smaller sub-claims that each focus on a specific component of the original statement, making it easier for a model to verify.
Begin each sub-claim with ###. If needed, refer to answers from earlier sub-claims using #1, #2, etc.
Decomposed claim:"""


prompt_plan = { 
    "gsm8k": """Please break down the question "{question}" into multiple specific sub-questions that address individual components of the original question. Use ### to mark the start for each sub-question.""",
    "TabMWP": """You have the following table:\n{table}\nFor the question "{question}", please break down the question into multiple specific sub-questions that address individual components of the original question, with the table as the reference. Use ### to mark the start of each sub-question.""",
    "convfinqa": """You have the following passages and table:\nPassages:\n{passage}\nTable:\n{table}\nPlease break down the question "{question}" into multiple specific sub-questions that address individual components of the original question, with the table and passages as the reference. Use ### to mark the start of each sub-question.""",
    "claim": prompt_plan_claim,
    "qa": prompt_plan_qa,
}

task_map = {
    "bamboogle": "qa", 
    "2wikimultihopqa": "qa", 
    "2wiki": "qa",
    "2wiki_origin": "qa",
    "m": "qa", 
    "hotpotqa": "qa", 
    "musique": "qa", 
    "hover": "claim", 
    "exfever": "claim"
}
# issues with 1.5B, missing base file
# tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True, trust_remote_code=True)

sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p, repetition_penalty=1.05, max_tokens=2048)

# Input the model name or path. Can be GPTQ or AWQ models.
model_path = args.model_path
# llm = LLM(model=model_path, tensor_parallel_size=args.tensor_parallel_size, gpu_memory_utilization=0.9, trust_remote_code=True)
llm = LLM(model=model_path, tokenizer=args.tokenizer, tensor_parallel_size=args.tensor_parallel_size, gpu_memory_utilization=0.9, trust_remote_code=True)

# Prepare your prompts
datasets = ["bamboogle", "2wikimultihopqa", "hotpotqa", "musique", "hover", "exfever","2wiki", "m", "2wiki_origin"]
if "," in args.datasets:
    datasets = args.datasets.split(",")
else:
    datasets = [args.datasets]
model_name = args.expname

ID_KEYS = ("question_id", "id", "_id", "qid", "uid")


def get_source_id(example):
    for key in ID_KEYS:
        if key in example and example[key] not in (None, ""):
            return example[key]
    return None



for dataset in datasets:
    prompts = []
    contexts = []
    q = []
    gold = []
    with open(f"eval_datasets/{dataset}/test_subsampled.jsonl", "r") as f:
        i = 0
        for lines in f:
            example = json.loads(lines)
            source_id = get_source_id(example)
            if task_map[dataset] == "claim":
                q.append(example["claim"])
                if example["label"] in ["SUPPORT", "SUPPORTED"]:
                    gold.append(["Yes"])
                else:
                    gold.append(["No"])

            else:
                tmp_answer = []
                for z in example["answers_objects"]:
                    tmp_answer += z["spans"]
                q.append(example["question_text"])
                gold.append(tmp_answer)
            contexts.append({
                "question_id": source_id if source_id is not None else i,
                "source_index": i,
            })
            i += 1
    for question, answer, item in zip(q, gold, contexts):
        item.update({"question": question, "answer": answer})
        prompt_tmp = copy.deepcopy(prompt_plan[task_map[dataset]])
        if task_map[dataset] == "qa":
            prompt_tmp = prompt_tmp.replace("{question}", question)
        elif task_map[dataset] == "claim":
            prompt_tmp = prompt_tmp.replace("{claim}", question)
        else:
            raise NotImplementedError
        prompts.append(
            [
                {"role": "user", "content": prompt_tmp.strip()}
            ] 
        )
    print("Dataset:", dataset, "Number of prompts:", len(prompts), "", len(contexts))
    examples = []
    os.makedirs(f"eval_datasets/test/{dataset}/prompts_decompose_test_t{args.temperature}_{model_name}", exist_ok=True)
    for i in trange(len(prompts)):
        if "qwen3" in args.expname:
            text = tokenizer.apply_chat_template(
                prompts[i],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
        else:
            text = tokenizer.apply_chat_template(
                prompts[i],
                tokenize=False,
                add_generation_prompt=True
            )
        print(text)
        N_samples = 1 if args.temperature == 0 else 3
        for j in range(N_samples):
            ctx = contexts[i]
            outputs = llm.generate([text], sampling_params)
            generated_text = outputs[0].outputs[0].text
            if j == 0:
                print(len(outputs))
                print('======\n', generated_text, '\n======')
            
            # replace with cheking for the matched results for prompt model
            decomposed_questions = []
            # Use regex to find all instances of ### Q<number>: <text>
            matches = re.findall(r"###\s*(Q\d+):\s*(.*)", generated_text)
            if not matches:
                    print(f"Error in decomposing (No matches found) \n\n {generated_text} \n\n")
                    decomposed_questions = "Error"
            else:
                for q_label, question_text in matches:
                    decomposed_questions.append({
                        "label": q_label.strip(), 
                        "text": question_text.strip(),
                        "needs_context": True
                    })
            ctx["source_index"] = i
            ctx["decompose_id"] = j
            ctx["decomposed"] = decomposed_questions
            examples.append(copy.deepcopy(ctx))
    if len(examples) > 0:
        with open(f"eval_datasets/test/{dataset}/prompts_decompose_test_t{args.temperature}_{model_name}/generate.jsonl", "w") as f:
            for example in examples:
                f.write(json.dumps(example) + "\n")
        examples = []
