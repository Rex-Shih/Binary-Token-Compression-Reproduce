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


def get_rouge_score(model, tokenizer, prompt, original_prompt, output, original_len, scorer, tao=0.9, Lambda=-0.01):
    with torch.no_grad():
        messages = [{"role": "user", "content": prompt}]
        encoded = tokenizer.apply_chat_template(messages, return_tensors="pt").cuda()  
        prompt_len = encoded.shape[1]          
        generated_ids = model.generate(encoded, max_new_tokens=30, do_sample=False)
        decoded = tokenizer.batch_decode(generated_ids)
        decoded = decoded[0][decoded[0].find("[/INST]")+len("[/INST]"):]
        output = output[output.find("[/INST]")+len("[/INST]"):]
        scores = scorer.score(decoded, output)

        rouge_l_score = scores['rougeL']
        if rouge_l_score.fmeasure >= tao:
            return 1-prompt_len/original_len
        return Lambda

# Load the dataset
dataset = load_dataset("yahma/alpaca-cleaned")
shuffled_dataset = list(dataset["train"].shuffle(seed=42))

train_data = shuffled_dataset[0:40000]
#val_data = shuffled_dataset[4:8]#0000:51700]
#test_data = shuffled_dataset[51700:]

dis_tokenizer = AutoTokenizer.from_pretrained('distilroberta-base')
dis_model = AutoModelForTokenClassification.from_pretrained("distilroberta-base", num_labels=2).cuda()
dis_model.classifier = nn.Sequential(
    nn.Linear(768, 4096), 
    nn.ReLU(),  
    nn.Linear(4096, 4096),  
    nn.ReLU(),
    nn.Linear(4096, 2) 
).cuda()
print(dis_model.classifier)
for param in dis_model.roberta.parameters():
    param.requires_grad = False


train_dataset = alphaca(train_data, dis_tokenizer)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)#, pin_memory=True, num_workers=4)
#val_dataset = alphaca(val_data, dis_tokenizer)
#val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, pin_memory=True, num_workers=4)

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

optimizer = torch.optim.AdamW(dis_model.parameters(), lr=3e-5)

print("finish building dataset!")
epochs = 2

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1").cuda()
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
criterion = nn.CrossEntropyLoss()


with open("./log3.txt", 'w') as logFile:
    for epoch in range(epochs):
        total_loss = 0
        for count, batch in tqdm(enumerate(train_loader)):
            
            optimizer.zero_grad()
            encoded_prompt, prompt, output, left_prompt = batch
            #logFile.write(str(encoded_prompt["attention_mask"]))
            #logFile.flush()
            
            
            encoded_prompt = {k:v.cuda() for k, v in encoded_prompt.items()}
            outputs = dis_model(**encoded_prompt)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            with torch.no_grad():
                detr_action = torch.argmax(probabilities, dim=-1)
                sample_action = []
                for j in range(4):
                    sample_action.append(torch.multinomial(probabilities.view(-1, 2), 1).view(probabilities.shape[:-1]))
                sample_reward = []
                for i in tqdm(range(detr_action.shape[0])):
                    detr_indices = [index for index, element in enumerate(detr_action[i, :]) if element == 1]
                    sample_prompts=[]
                    for j in range(4):
                        sample_indices = [index for index, element in enumerate(sample_action[j][i, :]) if element == 1]
                    
                        sample_prompt = dis_tokenizer.decode(encoded_prompt["input_ids"][i, sample_indices], skip_special_tokens=True)
                        sample_prompts.append(sample_prompt+left_prompt[i])

                    detr_prompt = dis_tokenizer.decode(encoded_prompt["input_ids"][i, detr_indices], skip_special_tokens=True)
                    detr_prompt = detr_prompt+left_prompt[i]
                    
                    messages = [{"role": "user", "content": prompt[i]}]
                    encoded = tokenizer.apply_chat_template(messages, return_tensors="pt").cuda()
                    
                    original_len = encoded.shape[1]
                    generated_ids = model.generate(encoded, max_new_tokens=30, do_sample=False)
                    output = tokenizer.batch_decode(generated_ids)
                    
                    detr_score = get_rouge_score(model, tokenizer, detr_prompt, prompt, output[0], original_len, scorer)
                    rewards = []
                    for j in range(4):
                        rewards.append(get_rouge_score(model, tokenizer, sample_prompts[j], prompt, output[0], original_len, scorer)-detr_score)
                    sample_reward.append(rewards)
            loss = 0
            
            for j in range(4):
                action = sample_action[j]
                action[encoded_prompt["attention_mask"] == 0] = -100
                reward = torch.tensor([i[j] for i in sample_reward])
                for k in range(outputs.logits.shape[0]):
                    neg_log_probs = criterion(outputs.logits[k, :, :], action[k, :], )
                    loss += neg_log_probs.cpu() * (reward[k])
                
                
                
            entropy = torch.distributions.Categorical(probabilities).entropy()
            entropy = entropy.mean()
            #print(entropy)
            loss = 1/4*loss - 0.001 * entropy
            loss.backward()
            optimizer.step()
            total_loss += reward.sum(dim=-1).item()
            
            if count%2==0:
                logFile.write(f"average reward of batch{count}: {total_loss/2/32}\n")
                logFile.flush()
                torch.save(dis_model.state_dict(), f"model_new/{epoch}-{count}.pt")
                total_loss = 0



