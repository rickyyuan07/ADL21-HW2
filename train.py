# -*- coding: utf-8 -*-
import json
import numpy as np
import random
import torch
from torch.utils.data import DataLoader, Dataset 
from transformers import AdamW, BertForQuestionAnswering, BertTokenizerFast, AutoModelForQuestionAnswering

from random import randint
from tqdm.auto import tqdm
from dataset import QA_Dataset

device = "cuda" if torch.cuda.is_available() else "cpu"

# Fix random seed for reproducibility
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
# same_seeds(0)

# Change "fp16_training" to True to support automatic mixed precision training (fp16)	
fp16_training = False
if fp16_training:
    from accelerate import Accelerator
    accelerator = Accelerator(fp16=True)
    device = accelerator.device

# Documentation for the toolkit:  https://huggingface.co/docs/accelerate/

"""## Load Model and Tokenizer
"""
model = AutoModelForQuestionAnswering.from_pretrained("wptoux/albert-chinese-large-qa").to(device)
tokenizer = BertTokenizerFast.from_pretrained("wptoux/albert-chinese-large-qa")

# You can safely ignore the warning message (it pops up because new prediction heads for QA are initialized randomly)

"""## Read Data

- Training set: 26935 QA pairs
- Dev set: 3523  QA pairs
- Test set: 3492  QA pairs

- {train/dev/test}_questions:	
  - List of dicts with the following keys:
   - id (int)
   - paragraph_id (int)
   - question_text (string)
   - answer_text (string)
   - answer_start (int)
   - answer_end (int)
   
   "id": "593f14f960d971e294af884f0194b3a7",
    "question": "舍本和誰的數據能推算出連星的恆星的質量？",
    "paragraphs": [
      2018,
      6952,
      8264,
      836
    ],
    "relevant": 836,
    "answer": {
      "text": "斯特魯維",
      "start": 108
    }

    "id": "7f4f68726faed6b987e348340a9e6a61",
    "question": "葉門是世界上經濟最落後的國家之一其主要倚賴什麼收入?",
    "paragraphs": [
      5262,
      7017,
      6952,
      49
    ]
- {train/dev/test}_paragraphs: 
  - List of strings
  - paragraph_ids in questions correspond to indexs in paragraphs
  - A paragraph may be used by several questions 
"""

def read_data(file, from_test=False):
    with open(file, 'r', encoding="utf-8") as reader:
        data = json.load(reader)
    with open('./data/context.json', 'r', encoding='utf-8') as fcontext:
        context = json.load(fcontext)
    questions = {}
    questions['question_text'] = [x['question'] for x in data]
    questions['id'] = [x['id'] for x in data]
    if not from_test:
        questions['paragraph_id'] = [x['relevant'] for x in data]
        questions['answer_text'] = [x['answer']['text'] for x in data]
        questions['answer_start'] = [x['answer']['start'] for x in data]
        questions['answer_end'] = [(x['answer']['start'] + len(x['answer']['text'])) for x in data]
        paragraphs = context
    else:
        questions['paragraph_id'] = list(range(len(data)))
        paragraphs = [x['paragraphs'] for x in data]
        paragraphs = [''.join([context[x] for x in p4]) for p4 in paragraphs]
    questions = [dict(zip(questions,t)) for t in zip(*questions.values())]
    return questions, paragraphs

train_questions, train_paragraphs = read_data("./data/train.json")
dev_questions, dev_paragraphs = read_data("./data/valid.json")
test_questions, test_paragraphs = read_data("./data/test.json", from_test=True)

from pprint import pprint
pprint(train_questions[0])
# pprint(train_paragraphs[0])
print('test')
pprint(test_questions[0])
pprint(test_paragraphs[0])
input()

"""## Tokenize Data"""
# Tokenize questions and paragraphs separately
# 「add_special_tokens」 is set to False since special tokens will be added when tokenized questions and paragraphs are combined in datset __getitem__ 

train_questions_tokenized = tokenizer([train_question["question_text"] for train_question in train_questions], add_special_tokens=False)
dev_questions_tokenized = tokenizer([dev_question["question_text"] for dev_question in dev_questions], add_special_tokens=False)
test_questions_tokenized = tokenizer([test_question["question_text"] for test_question in test_questions], add_special_tokens=False) 

train_paragraphs_tokenized = tokenizer(train_paragraphs, add_special_tokens=False)
dev_paragraphs_tokenized = tokenizer(dev_paragraphs, add_special_tokens=False)
test_paragraphs_tokenized = tokenizer(test_paragraphs, add_special_tokens=False)

train_set = QA_Dataset("train", train_questions, train_questions_tokenized, train_paragraphs_tokenized)
dev_set = QA_Dataset("dev", dev_questions, dev_questions_tokenized, dev_paragraphs_tokenized)
test_set = QA_Dataset("test", test_questions, test_questions_tokenized, test_paragraphs_tokenized)

train_batch_size = 16

# Note: Do NOT change batch size of dev_loader / test_loader !
# Although batch size=1, it is actually a batch consisting of several windows from the same QA pair
train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, pin_memory=True)
dev_loader = DataLoader(dev_set, batch_size=1, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False, pin_memory=True)

"""## Function for Evaluation"""

def evaluate(data, output):
    ##### TODO: Postprocessing #####
    answer = ''
    max_prob = float('-inf')
    num_of_windows = data[0].shape[1]
    a, b = 0, 0
    for k in range(num_of_windows):
        # Obtain answer by choosing the most probable start position / end position
        start_prob, start_index = torch.max(output.start_logits[k], dim=0)
        end_prob, end_index = torch.max(output.end_logits[k], dim=0)
        start_matrix, end_matrix = output.start_logits[k].unsqueeze(1), output.end_logits[k].unsqueeze(0)
        matrix = torch.matmul(start_matrix, end_matrix)
        print(tokenizer.decode(data[0][0][k]))
        # print(matrix.shape)
        # print(torch.argmax(matrix, keepdim=False) // matrix.shape[0], torch.argmax(matrix, keepdim=False) % matrix.shape[0])
        # print(start_index, end_index)
        if start_index >= end_index: # plus if start_index < [SEP] of question and paragraph
            s_p, s_i = torch.topk(output.start_logits[k], 2, dim=0)
            e_p, e_i = torch.topk(output.end_logits[k], 2, dim=0)
            s_p, s_i, e_p, e_i = s_p[1], s_i[1], e_p[1], e_i[1] 
            # print(f'r: {start_prob}, {s_p}, r: {end_prob}, {e_p}') # debug
            if s_p + end_prob > start_prob + e_p:
                start_prob, start_index = s_p, s_i
            else:
                end_prob, end_index = e_p, e_i

            if start_index >= end_index:
                continue

        # Probability of answer is calculated as sum of start_prob and end_prob
        prob = start_prob + end_prob
        
        # print(start_index, end_index)

        # Replace answer if calculated probability is larger than previous windows
        if prob > max_prob:
            max_prob = prob
            a, b = start_index, end_index
            # Convert tokens to chars (e.g. [1920, 7032] --> "大 金")
            answer = tokenizer.decode(data[0][0][k][start_index : end_index + 1])
    # print(answer, a, b)
    # if '[UNK]' in answer:
    #     print('evaluate:', answer, '->', end=' ')
    #     answer = answer.replace('[UNK]', '肏')
    #     print(answer)
    # Remove spaces in answer (e.g. "大 金" --> "大金")
    return answer.replace(' ','')

"""## Training"""
num_epoch = 3
validation = True
logging_step = 100
learning_rate = 1e-5
optimizer = AdamW(model.parameters(), lr=learning_rate)

if fp16_training:
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader) 

model.train()
print("Start Training ...")

for epoch in range(num_epoch):
    step = 1
    train_loss = train_acc = 0
    
    for data in tqdm(train_loader):	
        # Load all data into GPU
        data = [i.to(device) for i in data]
        
        # Model inputs: input_ids, token_type_ids, attention_mask, start_positions, end_positions (Note: only "input_ids" is mandatory)
        # Model outputs: start_logits, end_logits, loss (return when start_positions/end_positions are provided)  
        output = model(input_ids=data[0], token_type_ids=data[1], attention_mask=data[2], start_positions=data[3], end_positions=data[4])

        # Choose the most probable start position / end position
        start_index = torch.argmax(output.start_logits, dim=1)
        end_index = torch.argmax(output.end_logits, dim=1)
        
        # Prediction is correct only if both start_index and end_index are correct
        train_acc += ((start_index == data[3]) & (end_index == data[4])).float().mean()
        train_loss += output.loss
        
        if fp16_training:
            accelerator.backward(output.loss)
        else:
            output.loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()
        step += 1

        ##### TODO: Apply linear learning rate decay #####
        optimizer.param_groups[0]['lr'] -= learning_rate / (1684 * num_epoch)
        
        # Print training loss and accuracy over past logging step
        if step % logging_step == 0:
            print(f"Epoch {epoch + 1} | Step {step} | loss = {train_loss.item() / logging_step:.3f}, acc = {train_acc / logging_step:.3f}")
            train_loss = train_acc = 0

    if validation:
        print("Evaluating Dev Set ...")
        model.eval()
        with torch.no_grad():
            dev_acc = 0
            for i, data in enumerate(tqdm(dev_loader)):
                output = model(input_ids=data[0].squeeze(dim=0).to(device), token_type_ids=data[1].squeeze(dim=0).to(device),
                       attention_mask=data[2].squeeze(dim=0).to(device))
                # prediction is correct only if answer text exactly matches
                dev_acc += evaluate(data, output) == dev_questions[i]["answer_text"]
            print(f"Validation | Epoch {epoch + 1} | acc = {dev_acc / len(dev_loader):.3f}")
        model.train()

    # Save a model and its configuration file to the directory 「saved_model」 
    # i.e. there are two files under the direcory 「saved_model」: 「pytorch_model.bin」 and 「config.json」
    # Saved model can be re-loaded using 「model = BertForQuestionAnswering.from_pretrained("saved_model")」
    print("Saving Model ...")
    model_save_dir = f"saved_model_{epoch}" 
    model.save_pretrained(model_save_dir)

"""## Testing"""

from difflib import SequenceMatcher

def get_best_match(query, corpus, step=4, flex=3, case_sensitive=False, verbose=False):
    """Return best matching substring of corpus.

    Parameters
    ----------
    query : str
    corpus : str
    step : int
        Step size of first match-value scan through corpus. Can be thought of
        as a sort of "scan resolution". Should not exceed length of query.
    flex : int
        Max. left/right substring position adjustment value. Should not
        exceed length of query / 2.

    Outputs
    -------
    output0 : str
        Best matching substring.
    output1 : float
        Match ratio of best matching substring. 1 is perfect match.
    """

    def _match(a, b):
        """Compact alias for SequenceMatcher."""
        return SequenceMatcher(None, a, b).ratio()

    def scan_corpus(step):
        """Return list of match values from corpus-wide scan."""
        match_values = []

        m = 0
        while m + qlen - step <= len(corpus):
            match_values.append(_match(query, corpus[m : m-1+qlen]))
            if verbose:
                print(query, "-", corpus[m: m + qlen], _match(query, corpus[m: m + qlen]))
            m += step

        return match_values

    def index_max(v):
        """Return index of max value."""
        return max(range(len(v)), key=v.__getitem__)

    def adjust_left_right_positions():
        """Return left/right positions for best string match."""
        # bp_* is synonym for 'Best Position Left/Right' and are adjusted 
        # to optimize bmv_*
        p_l, bp_l = [pos] * 2
        p_r, bp_r = [pos + qlen] * 2

        # bmv_* are declared here in case they are untouched in optimization
        bmv_l = match_values[p_l // step]
        bmv_r = match_values[p_l // step]

        for f in range(flex):
            ll = _match(query, corpus[p_l - f: p_r])
            if ll > bmv_l:
                bmv_l = ll
                bp_l = p_l - f

            lr = _match(query, corpus[p_l + f: p_r])
            if lr > bmv_l:
                bmv_l = lr
                bp_l = p_l + f

            rl = _match(query, corpus[p_l: p_r - f])
            if rl > bmv_r:
                bmv_r = rl
                bp_r = p_r - f

            rr = _match(query, corpus[p_l: p_r + f])
            if rr > bmv_r:
                bmv_r = rr
                bp_r = p_r + f

            if verbose:
                print("\n" + str(f))
                print("ll: -- value: %f -- snippet: %s" % (ll, corpus[p_l - f: p_r]))
                print("lr: -- value: %f -- snippet: %s" % (lr, corpus[p_l + f: p_r]))
                print("rl: -- value: %f -- snippet: %s" % (rl, corpus[p_l: p_r - f]))
                print("rr: -- value: %f -- snippet: %s" % (rl, corpus[p_l: p_r + f]))

        return bp_l, bp_r, _match(query, corpus[bp_l : bp_r])

    if not case_sensitive:
        query = query.lower()
        corpus = corpus.lower()

    qlen = len(query)

    if flex >= qlen/2:
        print("Warning: flex exceeds length of query / 2. Setting to default.")
        flex = 3

    match_values = scan_corpus(step)
    pos = index_max(match_values) * step

    pos_left, pos_right, match_value = adjust_left_right_positions()

    return corpus[pos_left: pos_right].strip(), match_value

print("Evaluating Test Set ...")
import difflib
from nltk.util import ngrams

def get_best_match(query, corpus):
    ngs = ngrams( list(corpus), len(query) )
    ngrams_text = [''.join(x) for x in ngs]
    return difflib.get_close_matches(query, ngrams_text, n=1, cutoff=0)
result = []

from transformers import QuestionAnsweringPipeline

# the desired load model id
model_id = 2
model = AutoModelForQuestionAnswering.from_pretrained(f"saved_model_{model_id}").to(device)
model.eval()
with torch.no_grad():
    for i, data in enumerate(tqdm(test_questions)):
        nlp = QuestionAnsweringPipeline(model=model, tokenizer=tokenizer, task='question-answering', device=0)
        context = ''.join([test_paragraphs[x] for x in data['paragraph_id']])
        result.append(nlp(question=data['question_text'], context=context)['answer'])
        # output = model(input_ids=data[0].squeeze(dim=0).to(device), token_type_ids=data[1].squeeze(dim=0).to(device),
        #                attention_mask=data[2].squeeze(dim=0).to(device))
        # # result.append(evaluate(data, output))
        # ans = evaluate(data, output)
        # if '[UNK]' in ans:
        #     print(ans, '->', end=' ')
        #     ans = ans.replace('[UNK]', '?')
        #     # ans = get_best_match(ans, test_paragraphs[test_questions[i]['paragraph_id']], step=(len(ans) * 3)//4, flex=len(ans)//3)[0]
        #     ans = get_best_match(ans, test_paragraphs[test_questions[i]['paragraph_id']])[0]
        #     print(ans, i)
        # result.append(ans)
        
result_file = "result.csv"
with open(result_file, 'w') as f:	
    f.write("ID,Answer\n")
    for i, test_question in enumerate(test_questions):
    # Replace commas in answers with empty strings (since csv is separated by comma)
    # Answers in kaggle are processed in the same way
        f.write(f"{test_question['id']},{result[i].replace(',','')}\n")

print(f"Completed! Result is in {result_file}")

import textwrap
index = 15
print('\n'.join(textwrap.wrap(test_paragraphs[test_questions[index]['paragraph_id']], 60, break_long_words=True)))
print(test_questions[index]['question_text'])

with torch.no_grad():
    for i, data in enumerate(test_loader):
        if i == index:
            output = model(input_ids=data[0].squeeze(dim=0).to(device), token_type_ids=data[1].squeeze(dim=0).to(device),
                          attention_mask=data[2].squeeze(dim=0).to(device))
            print(evaluate(data, output))
            break