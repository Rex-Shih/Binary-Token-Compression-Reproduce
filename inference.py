from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch
import json
from rouge_score import rouge_scorer
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import sys

# Open a file to redirect stderr
with open('errorlog.txt', 'w') as f:
    # Save the original stderr
    original_stderr = sys.stderr
    # Redirect stderr to the file
    sys.stderr = f
    sys.stderr = original_stderr

class alphaca(Dataset):
    def __init__(self, dataset, tokenizer):
        #dataset = [json.loads(i) for i in dataset]
        self.prompt = [i['instruction']+' '+i['input'] for i in dataset]
        self.output = [i['output'] for i in dataset]
        self.encoded_prompt = tokenizer(self.prompt, return_tensors='pt', max_length=128, truncation=True, padding=True)
        self.left_prompt = []
        for i in range(len(self.prompt)):
            prompt=self.prompt[i]
            encoded = tokenizer.decode(self.encoded_prompt['input_ids'][i], skip_special_tokens=True)
            pos = prompt.find(encoded)
            
            if pos!=-1 and encoded!=prompt:
                left = prompt[pos+len(encoded):]
                if left == " ":
                    left = ""
                self.left_prompt.append(left)
            else:
                self.left_prompt.append("")
    def __len__(self):
        return len(self.prompt)
    def __getitem__(self, idx):
       
        return {k: v[idx] for k, v in self.encoded_prompt.items()}, self.prompt[idx], self.output[idx], self.left_prompt[idx]



# Load the dataset
dataset = load_dataset("yahma/alpaca-cleaned")
shuffled_dataset = list(dataset["train"].shuffle(seed=42))

#train_data = shuffled_dataset[0:40000]
val_data = shuffled_dataset[40000:41500]
#test_data = shuffled_dataset[51700:]

dis_tokenizer = AutoTokenizer.from_pretrained('distilroberta-base')
dis_model = AutoModelForTokenClassification.from_pretrained("distilroberta-base", num_labels=2).cuda()
dis_model.classifier = nn.Sequential(
    nn.Linear(768, 4096),  # First linear layer
    nn.ReLU(),  
    nn.Linear(4096, 4096),  # First linear layer
    nn.ReLU(),# ReLU activation
    nn.Linear(4096, 2)     # Second linear layer (output layer)
).cuda()
dis_model.load_state_dict(torch.load("model_new/0-1226.pt"))

#train_dataset = alphaca(train_data, dis_tokenizer)
#train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True, num_workers=4)
val_dataset = alphaca(val_data, dis_tokenizer)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

#optimizer = torch.optim.AdamW(dis_model.parameters(), lr=3e-5)

print("finish building dataset!")
#epochs = 2

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1").cuda()
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
criterion = nn.CrossEntropyLoss()


with open("./eval_model3_log.txt", 'w') as logFile:
    total_rouge = 0
    total_compression=0
    with torch.no_grad():
    #for epoch in range(epochs):
        sampled_prompt = []
        output_prompt = []    
        cool_score = []
        for count, batch in tqdm(enumerate(val_loader)):
            #optimizer.zero_grad()
            encoded_prompt, prompt, output_result, left_prompt = batch
            encoded_prompt = {k:v.cuda() for k, v in encoded_prompt.items()}
            outputs = dis_model(**encoded_prompt)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            sample_action= torch.multinomial(probabilities.view(-1, 2), 1).view(probabilities.shape[:-1])
            for i in tqdm(range(sample_action.shape[0])):
                sample_indices = [index for index, element in enumerate(sample_action[i, :]) if element == 1]
                sample_prompt = dis_tokenizer.decode(encoded_prompt["input_ids"][i, sample_indices], skip_special_tokens=True)
                sample_prompt = sample_prompt+left_prompt[i]
                sampled_prompt.append(sample_prompt) 
                messages = [{"role": "user", "content": prompt[i]}]
                encoded = tokenizer.apply_chat_template(messages, return_tensors="pt").cuda()
                
                original_len = encoded.shape[1]
                generated_ids = model.generate(encoded, max_new_tokens=30, do_sample=False)
                output = tokenizer.batch_decode(generated_ids)
                #output = output_result[i]
                
                messages = [{"role": "user", "content": sample_prompt}]
                encoded = tokenizer.apply_chat_template(messages, return_tensors="pt").cuda()  
                prompt_len = encoded.shape[1] 
                generated_ids = model.generate(encoded, max_new_tokens=30, do_sample=False)
                decoded = tokenizer.batch_decode(generated_ids)
                decoded = decoded[0][decoded[0].find("[/INST]")+len("[/INST]"):]
                output_prompt.append(decoded)
                output = output[0][output[0].find("[/INST]")+len("[/INST]"):]
                scores = scorer.score(decoded, output)

                rouge_l_score = scores['rougeL']
                total_rouge +=rouge_l_score.fmeasure
                cool_score.append(rouge_l_score.fmeasure) 
                total_compression+= prompt_len/original_len
            
            if count%10==0:
                
                logFile.write(f"average loss of rouge{count+1}: {total_rouge/(count+1)/32}, average comp_rate: {total_compression/(count+1)/32}\n")
                logFile.write(str(sampled_prompt)+'\n')
                logFile.write(str(output_prompt)+'\n')
                logFile.write(str(cool_score)+'\n')
                logFile.flush()
                
                
    logFile.write(f"final_rouge: {total_rouge/1500}, total_comp: {total_compression/1500}")



