# Inside-Out
Two Jointly Predictive Models for Word Representations and Phrase Representations

## Usage

### Requirements

To complile the souce codes, some external packages are required

* C++11
* Eigen
* OpenMP (for multithread)

### Input

#### Corpus
Each line of the corpus file represents a document in the corpus.
like:

```
... The cat sat on the mat. ...
... The quick brown fox jumps over the lazy dog. ...
```

#### Vocab Segmentation

Each line of the vocab segmentation file represents a vocab and its segmentations in the vocabulary.

Format:

```
word\t segmentation_1 segmentation_2 ...
```
like:

```
breakable	break able
```


### Run

```shell
./wordrep -train data.txt --mor_file vocab_seg.txt -word_output vec.txt -size 200 -window 5 -subsample 1e-4 -negative 10 -model being -binary 0 -iter 20
```

- -train, the input file of the corpus, each line a document;
- --mor_file, the input file of the vocab segmentation, each line represents the segmentations for a word;
- -word_output, the output file of the word embeddings;
- -binary, whether saving the output file in binary mode; the default is 0 (off);
- -word_size, the dimension of word embeddings; the default is 100;
- -window, max skip length between words; default is 5;
- -negative, the number of negative samples used in negative sampling; the deault is 5;
- -subsample, parameter for subsampling; default is 1e-4;
- -threads, the total number of threads used; the default is 1.
- -alpha, the starting learning rate; default is 0.025 for SEING and 0.05 for BEING; 
- -model, model used to learn the word embeddings; default is `being` (Continuous Bag of External and Internal Gram model) (use `seing` for Skip External and Internal Gram model)
- -min-count, the threshold for occurrence of words; default is 5;
- -iter, the number of iterations; default is 20;