# Generating-Reflections-Using-GPT-2-Few-Shot-Learning
 This repository is a demo of the Motivational Interviewing reflection generation using few-shot learning (code is tested for use with Huggingface library's pretrained GPT2 model). 

Few-shot learning refers to giving a pre-trained text-generation model (like GPT2) a few complete examples of the text generation task that we are trying to complete, and then providing an incomplete final example, omitting the part of the text that we want the text-generation model to produce. All of the examples are part of the input string that we provide to the model, with the final, incomplete example being the final example in that string. 

For our use case, each example consists of a prompt, response, and reflection. We format the input string so that each example is represented as follows:
`Prompt: [an example prompt] Response: [an example response] Reflection: [an example reflection]`.

We separate multiple examples with the newline character `\n`. 
Then, the final example contains a prompt, response, but the reflection is kept empty after the colon as follows:
`Prompt: [an example prompt] Response: [an example response] Reflection:` 

So, the final input to our text-generation model (GPT2) looks something like this: 

```
Prompt: [an example prompt] Response: [an example response] Reflection: [an example reflection] \n  

Prompt: [an example prompt] Response: [an example response] Reflection: [an example reflection] \n 

Prompt: [an example prompt] Response: [an example response] Reflection: [an example reflection] \n 

... 

Prompt: [an example prompt] Response: [an example response] Reflection:
```
Thus, the model attempts to replicate the style of the examples to fill in the blank portion in the final example. The number of examples as part of the input determines the number of 'shots', which we are referring to as the number of priming examples or primers. 

Note: This method is not training the model, so the model has no memory going from one prediction to the next using this method. You must use a pre-trained model for this task, the type and amount of data used in model pre-training will have an impact on the output as well, so we chose to use models pretrained on billions of words using the Huggingface library. Few-shot learning allows us to produce text in a specific style when we have a limited amount of labeled text data for the task. 


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

## Dataset 
The dataset (both the primers and samples) are examples taken from open-access Motivational Interviewing example transcripts. We extracted turns of conversations that included reflections and used those as primers. The samples are random snippets of example conversations between a therapist and a client. This dataset is only used to demonstrate how our script works and does not reflect the final performance of our reflection generation process.

## Issues you may have 
Depending on how much RAM you and your processor  the default model may not run or be very slow. The minimum amount of RAM recommended to run the gpt2-xl model is 16 gigabytes. If you have less, please try the gpt2-large or gpt2-medium models. The smaller models also run faster so if the inference time is too long, smaller models may help. 
