# Ensemble Judicial NER

 This repository contains an implementation for ensemble learning to do named entity recognition tasks on judicial data with **BiLSTM-CRF** architecture as *base learners*, using [**DeLFT**](https://github.com/kermitt2/delft) (**De**ep **L**earning **F**ramework for **T**ext, is a Keras and TensorFlow framework for text processing, focusing on sequence labelling, e.g.: named entity tagging, information extraction and text classification, e.g.: comment classification) with fundemantal changes to adapt it with our porpuse. The training and testing of this model was done on judicial data which can be found here: [***data/sequenceLabelling/CoNLL-2003***](data/sequenceLabelling/CoNLL-2003).

This *readme* file is divided into the following sections :
- Dataset description
- Dependencies installation
- Test running
- Train running

## Dataset description
This judicial dataset was annotated by *Gildas Tagny Ngompé et al. (2019)* (find at the bottom the paper to cite to use the dataset legitimately).

It contains 503 appeal court decisions and devided into three partitions (find them here [***data/sequenceLabelling/CoNLL-2003***](data/sequenceLabelling/CoNLL-2003)):

- Training set (*eng.train*): ~ 70%
- Validation set (*eng.testa*): ~ 10%
- Testing set (*eng.testb*): ~ 20%

A Kappa rates of 0.705 was obtained fornames entities, consequently the level of agreement is considered as substantial since it lies in the interval [0.61–0.80].
The dataset follows the *CoNLL-2003* format with the *BIEO* segment representation, and you can find 10 judicial specific classes (detailled distribution can be found in our paper):

- **NE001**: Person names
- **NE003**: Decision date
- **NE005**: Persons functions
- **NE006**: Legal citations
- **NE007**: Decisions cities
- **NE008**: Decisions Ids
- **NE009**: Jurisdictions

## Dependencies installation
After downloading the repo, it'd be better to create a virtual environement to execute the code (see tutorial [*here*](https://www.geeksforgeeks.org/creating-python-virtual-environment-windows-linux/)).
After that, you can run:
```pip install -r requirements.txt```

to install dependencies from the *requirements* directly.

## Test running
To reproduce the results described in our paper ***Ensemble BiLSTM-CRF vs CamemBERT: NER task on French court decisions*** run the following command:

> python3 nerRunner.py --model *model_name* eval

And replace *model_name* with the desired model or ensemble found in the paper:  **w2vec**, **fasttext**, **elmo**, **bert**, **judicial_camembert**, **e0** (BiLSTM-CRF models with static embeddings: fasttext and w2vec), **e1** (BiLSTM-CRF models with contextualized embeddings: elmo and bert) or **e2** (= e0 + e1).
To test the judicial camembert and **e3** (= e2 + judicial camembert) you need to download the 
[Judicial camembert](https://drive.google.com/drive/folders/1gDTHgEbSjcCmaZW17Qm-D6lHnycDsFLN?usp=sharing), put the *camembert* folder in the same directory with *nerRunner.py* file, and specify **judicial_camembert** or **e3** respictively.

[//]: # (However, Jud. CamemBERT still till now in private stage so you can neither rerun it nor the **e3**. Once published, we can provide information how to execute it.)

## Train running
If you want to retrain a model from scratch or use your own dataset (you should place it in [**data/sequenceLabelling/CoNLL-2003**](data/sequenceLabelling/CoNLL-2003)), you can do so by running this command:

> python3 nerTagger.py --architecture BidLSTM_CRF  --dataset-type conll2003  --embedding elmo-fr train_eval

You can use *k-cross-validation* training by adding the parameter *--fold-count k*. This will also download the embeddings *elmo-fr* if they are not found in ([***data/db***](data/db)), you can also change the desired emebdding as you need  , in this case you must add it in ```embedding-registry.json``` file.


*train_eval* parameter represent the desiered action to run, here we specified to do training then testing (evaluation), you can do just training by giving *train* as value.

Visit the original [DeLFT](https://github.com/kermitt2/delft) repo please to find more details about exection, commands and more.


**Paper to reference when using their dataset**:

*Gildas Tagny Ngompé et al. (2019)* (the dataset annotators):


```
@inproceedings{DBLP:conf/f-egc/NgompeHZMM18,
  author    = {Gildas Tagny Ngomp{\'{e}} and
               S{\'{e}}bastien Harispe and
               Guillaume Zambrano and
               Jacky Montmain and
               St{\'{e}}phane Mussard},
  editor    = {Bruno Pinaud and
               Fabrice Guillet and
               Fabien Gandon and
               Christine Largeron},
  title     = {Detecting Sections and Entities in Court Decisions Using {HMM} and
               {CRF} Graphical Models},
  booktitle = {Advances in Knowledge Discovery and Management - Volume 8 [Best of
               {EGC} 2017, Grenoble, France, and Best of {EGC} 2018, Paris, France]},
  series    = {Studies in Computational Intelligence},
  volume    = {834},
  pages     = {61--86},
  publisher = {Springer},
  year      = {2018},
  url       = {https://doi.org/10.1007/978-3-030-18129-1\_4},
  doi       = {10.1007/978-3-030-18129-1\_4},
  timestamp = {Thu, 14 Oct 2021 10:29:28 +0200},
  biburl    = {https://dblp.org/rec/conf/f-egc/NgompeHZMM18.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
