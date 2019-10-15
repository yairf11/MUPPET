# MUPPET
Code for the ACL 2019 paper 
[Multi-Hop Paragraph Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/1906.06606), 
by Yair Feldman and Ran El-Yaniv.

Much of the code in this repo is based on the following codebase: https://github.com/allenai/document-qa.

## Setup
First, clone the repository:
```
git clone https://github.com/yairf11/MUPPET.git
cd MUPPET
```

The easiest way to run this code is to use:
```
export PYTHONPATH=${PYTHONPATH}:`pwd`
```

### Dependencies
We require python >= 3.6 and tensorflow 1.9 (GPU version).
Other libraries can be installed using pip:
```
pip install -r requirements.txt
```

#### CoreNLP
Install Stanford CoreNLP using the instructions available [here](https://github.com/facebookresearch/DrQA#installing-drqa).

### Data
By default, we expect the word vectors to be stored in "~/data" and all other data to be stored in "./data". 
The expected file locations can be changed by altering config.py.
#### Word Vectors
The models we train use the common crawl 840 billion token GloVe word vectors from [here](https://nlp.stanford.edu/projects/glove/).
They are expected to exist in "~/data/glove/glove.840B.300d.txt". 
For example:
```
mkdir -p ~/data
mkdir -p ~/data/glove
cd ~/data/glove
wget http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip
rm glove.840B.300d.zip
```

#### Wikipedia
Download the Wikipedia dump:
```
mkdir -p data/wikipedia
cd data/wikipedia
wget https://nlp.stanford.edu/projects/hotpotqa/enwiki-20171001-pages-meta-current-withlinks-processed.tar.bz2
tar -xjvf enwiki-20171001-pages-meta-current-withlinks-processed.tar.bz2
```

Process Wikipedia to create an sqlite database and a TF-IDF model:
```
python hotpot/scripts/retriever/build_db.py data/wikipedia/enwiki-20171001-pages-meta-current-withlinks-processed/ data/db/wiki_hotpot_sentences_v1.1.db --num-workers 16
python hotpot/scripts/retriever/build_tfidf.py data/db/wiki_hotpot_sentences_v1.1.db data/db/ --num-workers 4
```

#### HotpotQA
Download the HotpotQA dataset:
```
mkdir data/hotpot
cd data/hotpot
wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json
wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json
wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json
wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_test_fullwiki_v1.json
```

Process the HotpotQA data:
```
python hotpot/data_handling/hotpot/build_hotpot_questions.py --num-workers 8
python hotpot/data_handling/hotpot/question_paragraph_scoring.py --num-workers 8
python hotpot/data_handling/hotpot/answer_span_detection.py
```

## Train
#### Encoder
Training an encoder model can be done as follows:
```
python hotpot/scripts/train_eval/ablate_iterative_hotpot.py <model name> --label-method br-as-cp --rank
```
However, this does not use ELMo embeddings. 
Instructions for training with ELMo embeddings will be added soon.

#### QA
Training a QA model can be done as follows:
``` 
python hotpot/scripts/train_eval/ablate_hotpot_qa.py <qa model dir>
```

## Evaluate
First, we find the top TF-IDF titles for each question in the dev/test set.
For example, for the top 64 titles, we could run:
``` 
python hotpot/scripts/data_building/build_hotpot_open_dataset.py question_test_64.json 64 --num-workers 16 
```

Then, we encode the documents corresponding to the top titles:
``` 
mkdir <encodings dir>
python hotpot/encoding/encode_documents.py <encodings dir>
<model dir> --questions_file question_test_64.json --checkpoint latest --ema --hotpot
```

Next, we perform a multi-hop MIPS retrieval using the encodings:
``` 
python hotpot/encoding/iterative_encoding_retrieval_batch.py <out dir> question_test_64.json <encodings dir> <model dir> --k1 8 --k2 45 --n1 32 --n2 64 --eval --rft --checkpoint latest --ema
```
Where n\<i> is the size of the search space for retrieval iteration \<i>.

Finally, we to run the QA model:
``` 
python hotpot/scripts/train_eval/hotpot_qa_distractors_eval.py <qa model dir> qa_pred.json -s latest -c retrieval_file -t 600 --input_file <out dir>/n2-64/n1-32/n1-32_n2-64_k1-8_k2-45.json --test_mode
```

Evaluate the results:
``` 
python hotpot_evaluate_v1.py qa_pred.json data/hotpot/hotpot_dev_fullwiki_v1.json
```

## Interactive Mode
Will be added soon.

## Pretrained Models
Will be added soon.