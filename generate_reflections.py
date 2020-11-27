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
    

def reflection_infer(text, model, tokenizer, device, max_len=200):
    tokenized_text = tokenizer.encode(text, return_tensors="pt")
    tokenized_text = tokenized_text.to(device)
    with torch.no_grad():

        summary_ids = model.generate(tokenized_text,
                                    max_length=tokenized_text.shape[1] + max_len,
                                    temperature=.75,
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


def generate_reflections(example_path, primers_path, model_tag, num_shot, results_filename, min_num_shot):
    
    # load df for primers
    prompt_df = pd.read_csv(primers_path)
    prompt_df=prompt_df.dropna(subset=['prompt','response','reflection'])
    print(f"columns priming: {list(prompt_df.columns)}")

    # load examples into dataframe
    fulldf = pd.read_csv(example_path)

    df = fulldf.sample(n=20, random_state=1)
    print(f"data columns: {list(df.columns)}")

    num_examples = len(df)
    

    # Optimize for inference
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    logging.getLogger("transformers").setLevel(logging.ERROR)
    results = []

    tokenizer = AutoTokenizer.from_pretrained(model_tag)
    model = AutoModelWithLMHead.from_pretrained(model_tag)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  
    model = model.to(device)
    prompt_set=prompt_df.copy()
  
    collectionofprompts=[prompt_set]

    for p, primes in enumerate(collectionofprompts):
        prompt_df=primes
        # print(f'Start primer set {promptnamedict[p]}')
        for j in range(min_num_shot, num_shot+1):
            for i in range(num_examples):
                # reload model
                start_time = time.time()
               
                prompt = gen_prompt_unique_primers(df, prompt_df, i, j)

                output = reflection_infer(prompt, model, tokenizer, device)
                reflection = first_reflection(output, j)

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
   
    # default paths to traning files, and output
    curpath = pathlib.Path(__file__).parent.absolute()
    dp = os.path.join(curpath, "./examples.csv")

    # parse command line arguments
    parser = argparse.ArgumentParser(description='Generate Simple Reflections')
    parser.add_argument('-p', type=str, default=dp,
                        help="Path to example prompt, response, reflection example csv file")
    parser.add_argument('-q', type=str, default=dp,
                        help="Path to priming set of prompt, response, reflection csv file")
    parser.add_argument('-minp', type=int, default=3,
                        help="min number of examples to feed network before inference")
    parser.add_argument('-maxp', type=int, default=3,
                        help="Maximum number of examples to feed network before inference")
    parser.add_argument('-m', type=str, default='gpt2-xl',
                        help="Model tag")
    parser.add_argument('-r', type=str, default='results',
                        help="results file name")
    args = parser.parse_args()

    generate_reflections(args.p, args.q, args.m, args.maxp, args.r, args.minp)

    #=========example commandline inputs ==================
    #python generate_reflections.py -m gpt2-medium -maxp 3 -minp 3 -q ./primers/filtered_primers.csv -p ./samples/filtered_data_reflections.csv -r example_results
    #python generate_reflections.py -m gpt2  -n 12 -p imtihan-reflections.tsv filtered_data_reflections.csv -r imtihan-results-more-shots.csv -r example_results.csv
