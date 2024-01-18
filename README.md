# Enhanced Abstractive Chat Summarization

## Overview

<p align="center">
  <img src="./sick 2.jpg" width="100%" height="80%">
</p>

This repository contains an enhanced implementation of the paper "Mind the Gap! Injecting Commonsense Knowledge for Abstractive Dialogue Summarization" by Seungone Kim et al. that can be found [here](https://github.com/SeungoneKim/SICK_Summarization).

Our work, titled "Leveraging Emojis, Keywords and Slang for Enhanced Abstractive Chat Summarization," builds upon the foundations laid by the aforementioned paper. We introduce novel extensions which consist in exploring the use of emojis, keywords, and an effective slang handling to improve the quality of generated summaries.

## Abstract

In this study, we present innovative enhancements to the abstractive chat summarization task. Our approach extends previous research that emphasized the advantages of injecting commonsense knowledge into dialogue summarization. The primary focus of our extensions includes:

1. **Emojis Significance:** We investigate the importance of emojis in dialogues and chat-like conversations. Emojis are explored as a rich source of information that can contribute to the generation of summaries with increased accuracy and contextual relevance.

2. **Keywords Injection:** We explore the impact of injecting keywords into the summarization process. Our findings highlight the beneficial role of keywords in improving the quality of dialogue summaries.

3. **Slang Handling:** We introduce a preprocessing technique to effectively handle slang in conversations. This addition aims to enhance the comprehensibility of generated summaries in the context of informal language use.

The results obtained from our framework show promising outcomes, indicating the potential for improved abstractive chat summarization. We believe that our contributions provide a valuable foundation for future research endeavors in this field.

## Setting
To utilize our enhanced abstractive chat summarization framework we suggest the use of Google Colab and the execution of the following steps.

Clone the repository:
```
git clone https://github.com/al3ssandrocaruso/Enhanced-Abstractive-Chat-Summarization.git
```
Run these commands:
```
!pip install -r requirements.txt
!sudo apt-get update -y
!sudo apt-get install python3.8
!sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1
!sudo update-alternatives --config python3
!sudo apt install python3-pip
!sudo apt install python3.8-distutils
!python -m spacy download en_core_web_sm
```

### Dataset Download
For training and evaluating on Samsum, we use dataset provided by [Hugging Face Dataset Library](https://github.com/huggingface/datasets). For Dialogsum, the dataset is not automatically provided, so you can download it from the url below,
```
https://drive.google.com/drive/folders/1plWw-jWvYo0QZsr1rv9BtXb0dYuRPsv1?usp=sharing
```
and put it under the directory of Enhanced-Abstractive-Chat-Summarization/data/DialogSum_Data .
```
mkdir data/DialogSum_Data
```

Also, you could download the preprocessed commonsense data from the url below,
```
https://drive.google.com/drive/folders/14Ot_3jYrXCONw_jUDgnCcojbFQUA10Ns?usp=sharing
```
and put it in the directory of Enhanced-Abstractive-Chat-Summarization/data/COMET_data.
```
mkdir data/COMET_data
```
To process the commonsense data [COMET-ATOMIC 2020](https://github.com/allenai/comet-atomic-2020) and [PARACOMET](https://github.com/skgabriel/paracomet) were used.

### Preprocessed JSON Files
You can download all the needed json files to run our extensions from here: 
```
TODO
```
and please put them inside this folder:
```
Enhanced-Abstractive-Chat-Summarization/data/COMET_data/paracomet/dialogue/
```
### Pretrained W2V
Although not strictly necessary, you can still download the W2V model (both the version trained on the twitter dataset (TODO:link to twitter dataset) and the one finetuned on Samsum) from here:
```
https://drive.google.com/drive/folders/1q8QOSHmAudSsRqEKhAu5fgcHKbsj8ooD?usp=sharing
``` 

## Train
To train the original SICK model execute the following command: 

```
!python3 ./Enhanced-Abstractive-Chat-Summarization/src/train_summarization_context.py --finetune_weight_path="./new_weights_sick" --best_finetune_weight_path="./new_weights_sick_best" --dataset_name="samsum" --use_paracomet=True --model_name="facebook/bart-large-xsum" --relation "xIntent" --epoch=1 --use_sentence_transformer True
```

In order to include our extensions please add the following parameters (singularly or as in supported combinations below):  

- emoji_m0 : If True emojis in the dataset are replaced with their aliases according to this dataset (TODO: link).
- emoji_m1 : If True it replaces emojis in the dataset with custom tokens containing their most similar words based on a W2V model which was trained on a twitter dataset and finetuned on Samsum dataset.
- keyword : If True KeyBert is used to build and add to the dataset new custom tokens containing the keywords it is capable to retrieve from each utterance. 
- slang : If True the model is trained on a dataset in which slang expressions are replaced with their corresponding actual meaning. 

As for now, the supported combinations of these parameters are: ```emoji_m1 + slang + keyword```, ```emoji_m1 + keyword```

*Note*: our implementations only work with Samsum dataset. 

We suggest to use different values for the ```--finetune_weight_path``` and ```--best_finetune_weight_path``` parameters on different runs to then be able to infer using all the models you trained by using the differently-named checkpoints (to be given as ```--model_checkpoint``` parameter to inference.py) 


## Inference
Obtain inferences executing the next command:
```
!python3 /content/Enhanced-Abstractive-Chat-Summarization/src/inference.py --dataset_name "samsum" --model_checkpoint="./new_weights_sick_best" --test_output_file_name="./summaries.txt" --use_paracomet True --num_beams 20 --train_configuration="full" --use_sentence_transformer True
```
Make sure to be using the right value for the ```--model_checkpoint``` parameter if you trained the model more than once using different extensions.

TODO: 
- share the missing jsons
- link the twitter dataset in the w2v
- link the emoji dataset for emoji_m0
