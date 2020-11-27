import time
import re
import os
import argparse
import pathlib
import logging
import pandas as pd
import gc
from transformers import AutoTokenizer, AutoModelWithLMHead
from sklearn.utils import shuffle
import torch


def gen_prompt(df, target_id, num_shot=0):
    prompt = ""
    for i in range(target_id-num_shot, target_id):
        if i < 0:
            i = i+len(df)

        # full example prompts
        prompt += "Prompt: " + df.iloc[i]['prompt'] + \
            "\nResponse: " + df.iloc[i]['response'] + \
            "\nReflection: " + df.iloc[i]['reflection'] + " \n\n"

    # test example partial prompt
    prompt += "Prompt: " + df.iloc[target_id]['prompt'] + \
        "\nResponse: " + df.iloc[target_id]['response'] + "\n"
    return prompt

def gen_prompt_unique_primers(df, primer_df, target_id, num_shot=0):
    primers = primer_df.sample(num_shot)
    prompt = ""
    for i,r in primers.iterrows():
        # full example prompts
        prompt += "Prompt: " + str(r['prompt']) + \
            "\nResponse: " + str(r['response']) + \
            "\nReflection: " + str(r['reflection']) + " \n\n"

    # test example partial prompt
    prompt += "Prompt: " + str(df.iloc[target_id]['prompt']) + \
        "\nResponse: " + str(df.iloc[target_id]['response']) + "\n"
    return prompt

def gen_prompt_primer_length_1std(df, target_id, num_shot=0):
    prompt = ""
    k = 0
    trials = 0
    while k < num_shot and trials < num_shot*3:
        sample = df.sample(n=1)
        while sample.index == target_id:
            sample = df.sample(n=1)
        toadd = "Prompt: " + sample.iloc[0]['prompt'] + \
            "\nResponse: " + sample.iloc[0]['response'] + \
            "\nReflection: " + sample.iloc[0]['reflection'] + " \n\n"
        if len(toadd) < 343 and len(toadd) > 163:
            prompt += toadd
            k += 1
        trials += 1
    
    if k < num_shot:
        # if it fails to find primers in the range, get random ones to remain
        vals = df.sample(n=num_shot-k)
        for i, r in vals.iterrows():
            prompt += "Prompt: " + df.iloc[i]['prompt'] + \
            "\nResponse: " + df.iloc[i]['response'] + \
            "\nReflection: " + df.iloc[i]['reflection'] + " \n\n"

    # test example partial prompt
    prompt += "Prompt: " + df.iloc[target_id]['prompt'] + \
        "\nResponse: " + df.iloc[target_id]['response'] + "\n"
    return prompt

def gen_prompt_manual_primers(df, target_id, num_shot=0):
    primers = [
"""Prompt: Great! Okay. Let's now chat about the bad things about smoking What is it about smoking that makes it bad?
Response: The smell, The health risks, The price. The feeling of smoking makes you outcast in some places.
Reflection: The cost, health risks, price, and perception are bad things about smoking.


""",
"""Prompt: Okay, so you associate Smell as something negative about smoking Please describe a time where you were worried about the smell of cigarettes but ended up smoking
Response:  My stepdad hates the smell of cigarettes so I do my best to try to avoid smelling like smoke even though I am a smoker. Mainly out of respect and understanding that it's not the best smell in the world.
Reflection: You think smoking has a bad smell and it affects the people around you.


""",
"""Prompt: Think back to the time when you were able to prevent yourself from smoking. What made it different from when you did smoke?
Response:  I really wanted to work that job at the time and smoke was not allowed there or while I was on company time. I used an e-cig to help with the cravings.
Reflection: Your desire to work helped you find an alternative to address your cravings.


""",
"""Prompt: Let me see if I recall correctly, so you think of Physical appearance as something bad about smoking Please describe a time where you thought about the effect of smoking on your appearance but ended up smoking
Response:  i worry about my skin aging faster due to smoking
Reflection: Skin aging is a concern of smoking for you.


""",
"""Prompt: Think back to the time when you were able to prevent yourself from smoking. What made it different from when you did smoke?
Response:  I had more of a positive and determined attitude. 
Reflection: Feeling positive and determined helped you to not smoke.


""",
"""Prompt: Please express a time where you experienced social stigma but still ended up smoking
Response:  When I was the only smoker amongst a group of friends on a night out. I smoked before we went into the theatre and they stood with me.
Reflection: You are worried about what your friends think of your smoking.


"""
    ]

    prompt= "".join(primers[:min(num_shot, 6)])
    prompt += "Prompt: " + df.iloc[target_id]['prompt'] + \
        "\nResponse: " + df.iloc[target_id]['response'] + "\n"
    return prompt

def reflection_infer(text, model, tokenizer, device, max_len=200):
    tokenized_text = tokenizer.encode(text, return_tensors="pt")
    tokenized_text = tokenized_text.to(device)
    with torch.no_grad():

        #  summary_ids = model.generate(tokenized_text,
        #                             max_length=tokenized_text.shape[1] + max_len,
        #                             # temperature=.175,
        #                             # repetition_penalty=1.3,
        #                             # bos_token_id=tokenizer.bos_token_id,
        #                             # early_stopping=True,
        #                             # top_k=1,
        #                             # top_p=0.8
        #                             )

        summary_ids = model.generate(tokenized_text,
                                    max_length=tokenized_text.shape[1] + max_len,
                                    temperature=.175,
                                    repetition_penalty=1.3,
                                    bos_token_id=tokenizer.bos_token_id,
                                    early_stopping=True,
                                    top_k=100,
                                    top_p=0.8)

    output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
   
    return output


def first_reflection(response, num_shot):
    lines = response.splitlines()

    response_lines = [line for line in lines if "Response: " in line]
    reflection_lines = [line for line in lines if "Reflection: " in line]

    if(len(reflection_lines) >= num_shot+1):
        reflect_line = reflection_lines[num_shot]
    elif len(lines) > lines.index(response_lines[num_shot]) + 1:
        reflect_line = lines[lines.index(response_lines[num_shot]) + 1]
    else:
        reflect_line = ''

    reflection = re.sub("Reflection: ", "", reflect_line)

    return reflection


def generate_reflections(example_path, primers_path, model_tag, num_shot, results_filename):
    # load df for primers
    prompt_df = pd.read_csv(primers_path)
    prompt_df=prompt_df.dropna(subset=['prompt','response','reflection'])
    print(f"columns priming: {list(prompt_df.columns)}")
    # load examples into dataframe
    fulldf = pd.read_csv(example_path)
    # df = shuffle(fulldf)

    df = fulldf.sample(n=20, random_state=1)
    print(f"data columns: {list(df.columns)}")

    num_examples = len(df)
    # num_examples = 20 + num_shot + 1
    

    # Optimize for inference
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    logging.getLogger("transformers").setLevel(logging.ERROR)
    results = []

    tokenizer = AutoTokenizer.from_pretrained(model_tag)
    model = AutoModelWithLMHead.from_pretrained(model_tag)
    # device = 'cpu'
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  
    device = torch.device('cuda:0')
    model = model.to(device)
    prompt_set=prompt_df.copy()

    # prompt0=prompt_set[prompt_set['edit_scores']==0]
    # prompt2=prompt_set[prompt_set['edit_scores']==0.2]
    # prompt4=prompt_set[prompt_set['edit_scores']==0.4]
    # prompt6=prompt_set[prompt_set['edit_scores']==0.6]
    # prompt8=prompt_set[prompt_set['edit_scores']==0.8]
    # prompt10=prompt_set[prompt_set['edit_scores']==1.0]
    # promptg2=prompt_set[prompt_set['edit_scores']>=0.2]
    # promptg4=prompt_set[prompt_set['edit_scores']>=0.4]
    # promptg6=prompt_set[prompt_set['edit_scores']>=0.6]
    # promptg8=prompt_set[prompt_set['edit_scores']>=0.8]
    #prompt0,prompt2,prompt4,prompt6,prompt8,prompt10,promptg2,promptg4,promptg6,promptg8,
    collectionofprompts=[prompt_set]
    # promptnamedict={0:'0.0',1:'0.2',2:'0.4',3:'0.6',4:'0.8',5:'1.0',6:'>=0.2',7:'>=0.4',8:'>=0.6',9:'>=0.8',10:'all'}

    for p, primes in enumerate(collectionofprompts):
        prompt_df=primes
        # print(f'Start primer set {promptnamedict[p]}')
        for j in range(4, num_shot+1):
            for i in range(num_examples):
                # reload model
                start_time = time.time()
                # model = None
                # tokenizer = AutoTokenizer.from_pretrained(model_tag)
                # model = AutoModelWithLMHead.from_pretrained(model_tag)
                # device = torch.device('cpu')  

                # generate prompt text to feed to the network
                # prompt = gen_prompt_primer_length_1std(df, i, j)
                prompt = gen_prompt_unique_primers(df, prompt_df, i, j)
                # prompt = gen_prompt(df, i, j)

                # prompt = gen_prompt_manual_primers(df, i, j)
                output = reflection_infer(prompt, model, tokenizer, device)
                reflection = first_reflection(output, j)

                # delete model to lower memory usage creep
                # del model
                # gc.collect()

                # create output spreadsheet row
                row = {'sample_id': i,
                    "num_shot": j,
                    # "primer_type":'all',
                    "prompt": df.iloc[i]['prompt'],
                    "response": df.iloc[i]['response'],
                    "reflection_gpt": reflection,
                    #    "reflection_human": df.iloc[i]['reflection']
                    }
                end_time = time.time()
                print(f'Model inference elapsed time: {end_time-start_time}')    
                print("{} out of {} completed at shot {}".format(i, num_examples, j))
                print(row)
                results.append(row)
                if i%5:
                    res_df = pd.DataFrame(results)
                    res_df.to_excel(results_filename + ".xlsx")
                    res_df.to_csv(results_filename + ".csv")


            # periodically save result data to files
            res_df = pd.DataFrame(results)
            res_df.to_excel(results_filename + ".xlsx")
            res_df.to_csv(results_filename + ".csv")


if __name__ == "__main__":

    # examples = 'imtihan-reflections.csv'
    # gpt2v = 'gpt2-xl'
    # n_shot = 3
    # results_file = 'imtihan_results.csv'

    # curpath = pathlib.Path(__file__).parent.absolute()
    # dp = os.path.join(curpath, "./" + examples)

    # generate_reflections(dp, gpt2v, n_shot, results_file)
   
    # # default paths to traning files, and output
    curpath = pathlib.Path(__file__).parent.absolute()
    dp = os.path.join(curpath, "./examples.csv")

    # parse command line arguments
    parser = argparse.ArgumentParser(description='Generate Simple Reflections')
    parser.add_argument('-p', type=str, default=dp,
                        help="Path to example prompt, response, reflection example csv file")
    parser.add_argument('-q', type=str, default=dp,
                        help="Path to priming set of prompt, response, reflection csv file")
    parser.add_argument('-n', type=int, default=3,
                        help="Maximum number of examples to feed network before inference")
    parser.add_argument('-m', type=str, default='gpt2-xl',
                        help="Model tag")
    parser.add_argument('-r', type=str, default='results',
                        help="results file name")
    args = parser.parse_args()

    generate_reflections(args.p, args.q, args.m, args.n, args.r)

    #python gpt.py -n 12 -p imtihan-reflections.tsv -r imtihan-results-more-shots.csv
    #python gpt.py -n 6 -p ./conversation_pairs/continuity_filtered_fahad_data.csv -q ./primers/imtihan-reflections.tsv -r filtered_data_reflections
    #python gpt.py -n 6 -p ./conversation_pairs/discontinuous_statements.csv -q ./primers/imtihan-reflections.tsv -r discontinuous_phrase_reflections
    #python gpt.py -n 6 -p ./conversation_pairs/continuity_filtered_fahad_data.csv -q ./primers/golden_primers.csv -r all_golden_primers_generated_data

