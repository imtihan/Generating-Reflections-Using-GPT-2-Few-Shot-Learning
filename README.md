# reflection_generation_demo
 This repository is a demo of the Motivational Interviewing reflection generation using few-shot learning (currently using GPT2 and few-shot learning). 

## Installation Instructions
To get started, please install the latest version of python. I recommend using the anaconda package for ease-of-use. 
You can download anaconda here: https://www.anaconda.com/products/individual

Before running the code, we need to install some packages.
Python package requirements are listed in requirements.txt
You can install the packages directly by running the following in a terminal: 

`pip install -r requirements.txt` 

Following that we install pytorch with the following (copy paste into the terminal and run):  
`pip install torch===1.7.0 torchvision===0.8.1 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html`

## How to run the code
Once the above packages are installed, we can start producing reflections by running the following in the terminal:

`python generate_reflections.py -m gpt2-medium -maxp 3 -minp 3 -q ./primers/filtered_primers.csv -p ./samples/filtered_data_reflections.csv -r example_results`

Each argument is preceded by the hyphen `-` character. The meaning of each argument is listed below:

`-p <filepath>` path to the file with the 'prompt' and 'response' as rows in a .csv file. These are the samples that will have the reflections generated on. Default is examples.csv

`-q <filepath>` path to the file with the priming examples. The file should be a .csv file containing rows with 'prompt', 'response', 'reflection' as rows. These are the prompts, examples to be fed to the model, that will be used for few-shot learning. 
Default is examples.csv

`-r <filepath>` path to the file and location where to save results. Default saves to results.csv in the same folder as the code file.

`-minp <number>` determines the minimum number of priming examples used as inputs for few-shot learning, default 3

`-maxp <number>` determines the maximum number of priming examples used as inputs for few-shot learning, default 3

`-m <model name>` declare the model from huggingface to be used for few shot learning, you can use any version of gpt that can be searched here, using the name the website describes: https://huggingface.co/models. Default is gpt2-xl


#### NOTE: The first time you run the code, the script will download the relevant models so the startup time will be much slower than subsequent runs of the code. Please ensure you have a stable internet connection as the models are a few gigabytes in size for the largest one.

## Issues you may have 
Depending on how much RAM you and your processor  the default model may not run or be very slow. The minimum amount of RAM recommended to run the gpt2-xl model is 16 gigabytes. If you have less, please try the gpt2-large or gpt2-medium models. The smaller models also run faster so if the inference time is too long, smaller models may help. 
