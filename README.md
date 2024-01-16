# Enhanced Abstractive Chat Summarization

## Overview

<p align="center">
  <img src="./sick 2.jpg" width="100%" height="80%">
</p>

This repository contains an enhanced implementation of the paper titled "Mind the Gap! Injecting Commonsense Knowledge for Abstractive Dialogue Summarization" by Seungone Kim. The original paper can be found [here](https://github.com/SeungoneKim/SICK_Summarization).

Our work, titled "Leveraging Emojis, Keywords and Slang for Enhanced Abstractive Chat Summarization," builds upon the foundations laid by the aforementioned paper. We introduce novel extensions to the task of abstractive chat summarization, exploring the use of emojis, keywords, and effective slang handling to improve the quality of generated summaries.

## Abstract

In this study, we present innovative enhancements to the abstractive chat summarization task. Our approach extends previous research that emphasized the advantages of injecting commonsense knowledge into dialogue summarization. The primary focus of our extensions includes:

1. **Emojis Significance:** We investigate the importance of emojis in dialogues and chat-like conversations. Emojis are explored as a rich source of information that can contribute to the generation of summaries with increased accuracy and contextual relevance.

2. **Keywords Injection:** We explore the impact of injecting keywords into the summarization process. Our findings highlight the beneficial role of keywords in improving the quality of dialogue summaries.

3. **Slang Handling:** We introduce a preprocessing technique to effectively handle slang in conversations. This addition aims to enhance the comprehensibility of generated summaries in the context of informal language use.

The results obtained from our framework show promising outcomes, indicating the potential for improved abstractive chat summarization. We believe that our contributions provide a valuable foundation for future research endeavors in this field.

## Built With

This project is built with the following major frameworks/libraries:

* [Python](https://www.python.org/) - Programming language used for implementation.

## Usage

### Download Dataset

To download the dataset, run the following script:

```

```

## Getting Started

To utilize our enhanced abstractive chat summarization framework, follow these steps:

1. Clone the repository:

   ```
   git clone https://github.com/al3ssandrocaruso/Enhanced-Abstractive-Chat-Summarization.git
   ```

2. Install the required packages:

   ```
   pip install -r requirements.txt
   ```


### Train SICK Model

To train the SICK model, use the following command:

```
python prova.py
```

### Get Inferences
Obtain inferences using the following script:

```
python prova.py
```


# Our implementations
Pretrained model V2V emoji on twitter data: https://drive.google.com/drive/folders/1iIiuXpx4UYdNLhhGkhymsUa1n6tE6HxT?usp=sharing

Some additional configs to specify.

- emoji_m0 : If True uses an emoji remapping, more information in the folder _emoji/PreprocessJSON_Emoji.ipynb_".
- emoji_m1 : If True uses an emoji remapping with a word2vector model, more information in the folder _emoji/W2V_Emoji.ipynb_".
- keyword : If True uses KeyBert to get the most relevant word in the sentence, more information in the folder _keyword/KeyBERT.ipynb_".
- slang : If True remap phrases's slang, more information in the folder _utils/slang_conversion.ipynb_.


# Mind the Gap! Injecting Commonsense Knowledge for Abstractive Dialogue Summarization
The official repository for the paper "Mind the Gap! Injecting Commonsense Knowledge for Abstractive Dialogue Summarization" accepted at COLING 2022.

Paper Link : https://arxiv.org/abs/2209.00930

Youtube Explanation : https://www.youtube.com/watch?v=xHr3Ujlib4Y

Overview of method, SICK (Summarizing with Injected Commonsense Knowledge).
<p align="center">
  <img src="./SICK_overview.png" width="100%" height="80%">
</p>

## Setting
The following command will clone the project:
```
git clone https://github.com/SeungoneKim/SICK_Summarization.git
```

Before experimenting, you can make a virtual environment for the project.
```
conda create -n sick python=3.8
conda activate sick
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 -f https://download.pytorch org/whl/cu113/torch_stable.html
pip install -r requirements.txt
pip install -U spacy
python -m spacy download en_core_web_sm
```

## Dataset Download
For training and evaluating on Samsum, we use dataset provided by [Hugging Face Dataset Library](https://github.com/huggingface/datasets). For Dialogsum, the dataset is not automatically provided, so you can download it from the url below,
```
https://drive.google.com/drive/folders/1CuZaU5Xw0AiIPaBTRrjToFkktS7_6KwG?usp=share_link
```
and put it under the directory of SICK_summarization/data/DialogSum_Data.
```
mkdir data/DialogSum_Data
```

Also, you could download the preprocessed commonsense data from the url below,
```
https://drive.google.com/drive/folders/1z1MXBGJ3pt0lC5dneMfFrQgxXTD8Iqrr?usp=share_link
```
and put it under the directory of SICK_summarization/data/COMET_data.
```
mkdir data/COMET_data
```

To process the commonsense data, we used [COMET-ATOMIC 2020](https://github.com/allenai/comet-atomic-2020) and [PARACOMET](https://github.com/skgabriel/paracomet) github repository. Huge thanks to the authors for providing the awesome code:)


## Training & Evaluation
You can use the following commands to train either SICK, SICK++. Note that for SICK++, we use the customized model class, BartForConditionalGeneration_DualDecoder and the customized trainer class DualDecoderTrainer each located in models/bart.py and src/trainer.py. It will automatically loaded if you set the following arguments correctly.


For training SICK, you can use the following command.
```
CUDA_VISIBLE_DEVICES="1" python3 train_summarization_context.py --finetune_weight_path="./new_weights_sick" --best_finetune_weight_path="./new_weights_sick_best" --dataset_name="samsum" --use_paracomet=True --model_name="facebook/bart-large-xsum" --relation "xIntent" --epoch=1 --use_sentence_transformer True
```
Some configs to specify.

- dataset_name : Specify either "samsum" or "dialogsum"
- use_paracomet : If you set to true, it will use paracomet, and if not, it will use comet by default.
- use_sentence_transformer : If you would like to use the commonsense selected with sentence_transformer, you should use this argument
- use_roberta : If you would like to use the commonsense selected with roberta_nli model, you should use this argument.
- relation : If you would only like to use one of the 5 possible relations, you could specify it with this argument.


For training SICK++, you can use the following command.
```
CUDA_VISIBLE_DEVICES="1" python3 train_summarization_full.py --finetune_weight_path="./new_weights_sickplus" --best_finetune_weight_path="./new_weights_sickplus_best" --dataset_name="samsum" --use_paracomet=True --model_name="facebook/bart-large-xsum" --relation "xIntent" --supervision_relation "xIntent" --epoch=1 --use_sentence_transformer True
```
Some configs to specify.

- dataset_name : Specify either "samsum" or "dialogsum"
- use_paracomet : If you set to true, it will use paracomet, and if not, it will use comet by default.
- use_sentence_transformer : If you would like to use the commonsense selected with sentence_transformer, you should use this argument
- use_roberta : If you would like to use the commonsense selected with roberta_nli model, you should use this argument.
- relation : If you would only like to use one of the 5 possible relations, you could specify it with this argument.
- supervision_relation : If you would only like to use one of the 5 possible supervision relations, you could specify it with this argument.


For inferencing either SICK or SICK++, you could use the following command.
```
CUDA_VISIBLE_DEVICES="1" python3 inference.py --dataset_name "samsum" --model_checkpoint="./new_weights_paracomet_best" --test_output_file_name="./tmp_result.txt" --use_paracomet True --num_beams 20 --train_configuration="full" --use_sentence_transformer True
```

Some configs to specify.
- dataset_name : Specify either "samsum" or "dialogsum"
- use_paracomet : If you set to true, it will use paracomet, and if not, it will use comet by default.
- use_sentence_transformer : If you would like to use the commonsense selected with sentence_transformer, you should use this argument
- use_roberta : If you would like to use the commonsense selected with roberta_nli model, you should use this argument.
- relation : If you would only like to use one of the 5 possible relations, you could specify it with this argument.
- supervision_relation : If you would only like to use one of the 5 possible supervision relations, you could specify it with this argument.
- num_beams : beam size during decoding
- model_checkpoint : The checkpoint to use during inference(test) time. We suggest to use the best_finetune_weight_path during training.
- training_configuration : If you want to test SICK, type "context". If you want to test SICK++, type "full". 


## Citation
If you find this useful, please consider citing our paper:
```
@inproceedings{kim2022mind,
  title={Mind the Gap! Injecting Commonsense Knowledge for Abstractive Dialogue Summarization},
  author={Kim, Seungone and Joo, Se June and Chae, Hyungjoo and Kim, Chaehyeong and Hwang, Seung-won and Yeo, Jinyoung},
  booktitle={Proceedings of the 29th International Conference on Computational Linguistics},
  pages={6285--6300},
  year={2022}
}
```  

## Point of contact
For any questions about the implementation or content of the paper, you could contact me via the following email:)
```
louisdebroglie@kaist.ac.kr
```
