import os
import numpy as np
from delft.sequenceLabelling import Sequence
from delft.utilities.Tokenizer import tokenizeAndFilter
from delft.utilities.Embeddings import Embeddings,test
from delft.utilities.Utilities import stats
from delft.sequenceLabelling.reader import load_data_and_labels_xml_file, load_data_and_labels_conll, load_data_and_labels_lemonde, load_data_and_labels_ontonotes
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, f1_score
from numpy import nditer, mean
import keras.backend as K
import argparse
import time
from NERDA.datasets import get_conll_data
from NERDA.models import NERDA

tag_scheme = [            'B_NE001',
                            'B_NE003',
                            'B_NE005',
                            'B_NE006',
                            'B_NE007',
                            'B_NE008',
                            'B_NE009',
                            'I_NE001',
                            'I_NE003',
                            'I_NE005',
                            'I_NE006',
                            'I_NE009',
]
tag_outside='O'
# hyperparameters for network
dropout = 0.1
# hyperparameters for training camembert
training_hyperparameters = {
'epochs' : 14,
'warmup_steps' : 700,
'train_batch_size': 32,
'learning_rate': 0.0001
}
def voting(ys=[],coeffs=[],coef_alltime=False):
    unique_ys = list(set(ys))#no001, noo1, noo2, n002
    # no001,noo2
    if len(unique_ys) > 1:
        maximum = ''
        max_occ = 1
        for unique_y in unique_ys:
            temp_occ = ys.count(unique_y)
            if temp_occ > max_occ:
                max_occ = temp_occ
                maximum = unique_y
        if max_occ > len(ys)//2:
            if not coef_alltime:
                return maximum
            else:
                max_coefs = [0 for x in range(len(unique_ys))]
                for i,unique_y in enumerate(unique_ys):
                    indices = [i for i, x in enumerate(ys) if x == unique_y]
                    for j in indices:
                        max_coefs[i] += coeffs[j]
                max_coef_index = max_coefs.index(max(max_coefs))
                return unique_ys[max_coef_index]
        else:
            max_coefs = [0 for x in range(len(unique_ys))]
            for i,unique_y in enumerate(unique_ys):
                indices = [i for i, x in enumerate(ys) if x == unique_y]
                for j in indices:
                    max_coefs[i] += coeffs[j]
            max_coef_index = max_coefs.index(max(max_coefs))
            print(f'conflict max(max_coefs):{max(max_coefs)}\n{*ys,}\t{unique_ys[max_coef_index]}')
            return unique_ys[max_coef_index]
    else : return ys[0]

    
# usual eval on CoNLL 2003 eng.testb 
def eval(model,
         dataset_type='conll2003', 
         lang='en',
         architecture='BidLSTM_CRF', 
         use_ELMo=False, 
         use_BERT=False, 
         data_path=None,): 

    print('Loading CoNLL-2003 NER data...')
    x_test, y_test = load_data_and_labels_conll('data/sequenceLabelling/CoNLL-2003/eng.testb')
    x_test_list, y_test_list = x_test.tolist(), y_test.tolist()
    
    new_y_test = []
    for y in y_test_list:
        temp = []
        for p_y in y:
            temp.append(p_y.replace('E_', 'I_'))
        new_y_test.append(temp)

    print(f'x_test shape {x_test.shape}\n \
            y_test shape {y_test.shape}')
    stats(x_eval=x_test, y_eval=y_test)

    
    coeffs = []
    y_predecteds = []
    print(f'======== {model}')
    model_name1 = 'ner-en-conll2003-BidLSTM_CRF-wiki.fr'
    model_name2 = 'ner-en-conll2003-BidLSTM_CRF-elmo-fr-embeddings-11epocs_3rd-run_2-folds'
    model_name3 = 'ner-en-conll2003-BidLSTM_CRF-bert-base-en-emebddings-11epocs'
    model_name4 = 'ner-en-conll2003-BidLSTM_CRF-glove-840B-embeddings-11epocs'
    model_name5 = 'ner-en-conll2003-BidLSTM_CRF-word2vec-embeddings-25epocs'
    # model_name2 = 'ner-en-conll2003-BidLSTM_CRF-elmo-fr-embeddings-11epocs'
    # model_name4 = 'ner-en-conll2003-BidLSTM_CRF-wiki.fr-embeddings-11epocs'
    if model == 'camembert':
        camembert_model = NERDA(device="cuda:0",
                            tag_scheme = tag_scheme,
                            tag_outside = tag_outside,
                            transformer = '../camembert/camembert_jurica_512/',
                            )
        start_time = time.time()
        print("\nEvaluation on test set:")
        y_predecteds.append(camembert_model.predict(x_test_list))
    elif model == 'w2vec':
        model = Sequence(model_name5)
        model.load()
        start_time = time.time()
        print("\nEvaluation on test set:")
        y_predecteds.append(model.eval(x_test, y_test))
        coeffs.append(1)
    elif model == 'fasttext':
        model = Sequence(model_name1)
        model.load()
        start_time = time.time()
        print("\nEvaluation on test set:")
        y_predecteds.append(model.eval(x_test, y_test))
        coeffs.append(1)
    elif model == 'elmo':
        model = Sequence(model_name2)
        model.load()
        start_time = time.time()
        print("\nEvaluation on test set:")
        y_predecteds.append(model.eval(x_test, y_test))
        coeffs.append(1)
    elif model == 'bert':
        model = Sequence(model_name3)
        model.load()
        start_time = time.time()
        print("\nEvaluation on test set:")
        y_predecteds.append(model.eval(x_test, y_test))
        coeffs.append(1)
    elif model == 'e0':
        model5 = Sequence(model_name5)
        model1 = Sequence(model_name1)
        model1.load()
        model5.load()
        start_time = time.time()
        print("\nEvaluation on test set:")
        y_predecteds.append(model1.eval(x_test, y_test))
        y_predecteds.append(model5.eval(x_test, y_test))
        coeffs = [.4,.6]
    elif model == 'e1':
        model2 = Sequence(model_name2)
        model3 = Sequence(model_name3)
        model2.load()
        model3.load()
        start_time = time.time()
        print("\nEvaluation on test set:")
        y_predecteds.append(model2.eval(x_test, y_test))
        y_predecteds.append(model3.eval(x_test, y_test))
        coeffs = [.6,.4]
    elif model == 'e2':
        model1 = Sequence(model_name1)
        model5 = Sequence(model_name5)
        model2 = Sequence(model_name2)
        model3 = Sequence(model_name3)
        model1.load()
        model5.load()
        model2.load()
        model3.load()
        start_time = time.time()
        print("\nEvaluation on test set:")
        y_predecteds.append(model1.eval(x_test, y_test))
        y_predecteds.append(model5.eval(x_test, y_test))
        y_predecteds.append(model2.eval(x_test, y_test))
        y_predecteds.append(model3.eval(x_test, y_test))
        coeffs = [.3,.3, .2,.2]
    else:
        print('unknown model')
        return
    print(f'coeffs {coeffs}')


    with open('y.txt', 'w') as f:
        y_voted = []
        ys = []
        if len(y_predecteds) == 1:
            f.write(f'X_test\t\tY_test\t\tModel\n')#\t\tGlove
            for x, y, y1 in \
            zip(x_test, y_test, \
            y_predecteds[0]):
                for p_x, p_y, p_y1 in zip(x, y,y1):
                    ys_transformed = []
                    for tr in [ p_y1]:
                        if tr == '<PAD>' or tr == 'O':
                            ys_transformed.append('NE000')
                        else:
                            ys_transformed.append(tr)
                    #[p_y1, p_y2, p_y3, p_y4, p_y5],\
                    y_v = p_y1
                    if y_v == '<PAD>':
                        y_v= 'NE000'
                    y_voted.append(y_v)
                    ys.append(p_y)
                    f.write(f'{p_x}\t\t{p_y}\t\t{p_y1}\t\t{y_v}\n')
        elif len(y_predecteds) == 2:
            f.write(f'X_test\t\tY_test\t\tModel1\t\tModel2\n')#\t\tGlove
            for x, y, y1,y5 in \
            zip(x_test, y_test, \
            y_predecteds[0], y_predecteds[1]):
                for p_x, p_y, p_y1,p_y5 in zip(x, y,y1,y5):
                    ys_transformed = []
                    for tr in [ p_y1,p_y5]:
                        if tr == '<PAD>' or tr == 'O':
                            ys_transformed.append('NE000')
                        else:
                            ys_transformed.append(tr)
                    #[p_y1, p_y2, p_y3, p_y4, p_y5],\
                    y_v = voting(  ys=ys_transformed, \
                                            coeffs=coeffs )
                    if y_v == '<PAD>':
                        y_v= 'NE000'
                    y_voted.append(y_v)
                    ys.append(p_y)
                    f.write(f'{p_x}\t\t{p_y}\t\t{p_y1}\t\t{p_y5}\t\t{y_v}\n')
        elif len(y_predecteds) == 4:
            f.write(f'X_test\t\tY_test\t\tModel1\t\tModel2\t\tModel3\t\tModel4\n')#\t\tGlove
            for x, y, y1,y2,y3,y5 in \
            zip(x_test, y_test, \
            y_predecteds[0], y_predecteds[1], y_predecteds[2], y_predecteds[3]):
                for p_x, p_y, p_y1, p_y2, p_y3,p_y5 in zip(x, y,y1, y2, y3,y5):
                    ys_transformed = []
                    for tr in [ p_y1, p_y2, p_y3,p_y5]:
                        if tr == '<PAD>' or tr == 'O':
                            ys_transformed.append('NE000')
                        else:
                            ys_transformed.append(tr)
                    #[p_y1, p_y2, p_y3, p_y4, p_y5],\
                    y_v = voting(  ys=ys_transformed, \
                                            coeffs=coeffs )
                    if y_v == '<PAD>':
                        y_v= 'NE000'
                    y_voted.append(y_v)
                    ys.append(p_y)
                    f.write(f'{p_x}\t\t{p_y}\t\t{p_y1}\t\t{p_y2}\t\t{p_y3}\t\t{p_y5}\t\t{y_v}\n')
        
    
    runtime = round(time.time() - start_time, 3)
    # print(f'Diasappear {set(ys) - set(y_voted)}')
    metrics = precision_recall_fscore_support(ys, y_voted, average=None,)
    metrics2 = f1_score(ys, y_voted, average=None,)
    # print(f'\nmetrics all in one {metrics}\n\n')
    print('============== Voting scores =============\n\tprecision\trecall\tfscore\tsupport\n')
    for label, precision, recall, fscore, support in zip(list(set(ys)),
                                                        nditer(metrics[0]),\
                                                        nditer(metrics[1]), \
                                                        nditer(metrics[2]), \
                                                        nditer(metrics[3]),):
        print(f'{label}\t\t{round(float(precision),4)*100}\t\t{round(float(recall),4)*100}\t\t{round(float(fscore),4)*100}\t\t{support}')
    print('==========================================')
    print(f'AVG:\t{mean(metrics[0])}\t{mean(metrics[1])}\t{mean(metrics[2])}\t{mean(metrics[3])}')

    print('============== Voting scores (2) =============\n\tf1_score\n')
    for label, fscore in zip(list(set(ys)), \
                            nditer(metrics[0]),):
        print(f'{label}\t\t{round(float(fscore),4)*100}')
    print('==========================================')
    print(f'AVG:\t{mean(metrics[0])}')

    print("runtime: %s seconds " % (runtime))

if __name__ == "__main__":

    architectures = ['BidLSTM_CRF', 'BidLSTM_CNN_CRF', 'BidLSTM_CNN_CRF', 'BidGRU_CRF', 'BidLSTM_CNN', 'BidLSTM_CRF_CASING', 
                     'bert-base-en', 'bert-large-en', 'scibert', 'biobert']

    parser = argparse.ArgumentParser(
        description = "Neural Named Entity Recognizers")

    parser.add_argument("action", help="one of [train, train_eval, eval, tag]")
    parser.add_argument("--fold-count", type=int, default=1, help="number of folds or re-runs to be used when training")
    parser.add_argument("--lang", default='en', help="language of the model as ISO 639-1 code")
    parser.add_argument("--dataset-type",default='conll2003', help="dataset to be used for training the model")
    parser.add_argument("--train-with-validation-set", action="store_true", help="Use the validation set for training together with the training set")
    parser.add_argument("--architecture",default='BidLSTM_CRF', help="type of model architecture to be used, one of "+str(architectures))
    parser.add_argument("--use-ELMo", action="store_true", help="Use ELMo contextual embeddings") 
    parser.add_argument("--use-BERT", action="store_true", help="Use BERT extracted features (embeddings)") 
    parser.add_argument("--data-path", default=None, help="path to the corpus of documents for training (only use currently with Ontonotes corpus in orginal XML format)") 
    parser.add_argument("--file-in", default=None, help="path to a text file to annotate") 
    parser.add_argument("--file-out", default=None, help="path for outputting the resulting JSON NER anotations") 
    parser.add_argument("--model", default=None, help="model or ensemble you want to use") 
    parser.add_argument(
        "--embedding", default=None,
        help=(
            "The desired pre-trained word embeddings using their descriptions in the file"
            " embedding-registry.json."
            " Be sure to use here the same name as in the registry ('glove-840B', 'fasttext-crawl', 'word2vec'),"
            " and that the path in the registry to the embedding file is correct on your system."
        )
    )

    args = parser.parse_args()

    action = args.action
    model = args.model
    if action not in ('train', 'tag', 'eval', 'train_eval'):
        print('action not specifed, must be one of [train, train_eval, eval, tag]')
    lang = args.lang
    dataset_type = args.dataset_type
    train_with_validation_set = args.train_with_validation_set
    use_ELMo = args.use_ELMo
    use_BERT = args.use_BERT
    architecture = args.architecture
    if architecture not in architectures and architecture.lower().find("bert") == -1:
        print('unknown model architecture, must be one of', architectures)
    data_path = args.data_path
    file_in = args.file_in
    file_out = args.file_out

    # name of embeddings refers to the file embedding-registry.json
    # be sure to use here the same name as in the registry ('glove-840B', 'fasttext-crawl', 'word2vec'), 
    # and that the path in the registry to the embedding file is correct on your system
    # below we set the default embeddings value
    if args.embedding is None:
        embeddings_name = 'glove-840B'
        if lang == 'en':
            if dataset_type == 'conll2012':
                embeddings_name = 'fasttext-crawl'
        elif lang == 'fr':
            embeddings_name = 'wiki.fr'
    else:
        embeddings_name = args.embedding

    if action == 'train':
        train(embeddings_name, 
            dataset_type, 
            lang, 
            architecture=architecture, 
            use_ELMo=use_ELMo,
            use_BERT=use_BERT,
            data_path=data_path)

    if action == 'train_eval':
        import tensorflow as tf
        strategy = tf.distribute.MirroredStrategy( ["GPU:0", "GPU:1"] )
        with strategy.scope():
            if args.fold_count < 1:
                raise ValueError("fold-count should be equal or more than 1")
            train_eval(embeddings_name, 
                dataset_type, 
                lang, 
                architecture=architecture, 
                fold_count=args.fold_count, 
                train_with_validation_set=train_with_validation_set, 
                use_ELMo=use_ELMo,
                use_BERT=use_BERT,
                data_path=data_path)

    if action == 'eval':
        eval(model, dataset_type, lang, architecture=architecture, use_ELMo=use_ELMo, use_BERT=use_BERT)

    if action == 'tag':
        if lang is not 'en' and lang is not 'fr':
            print("Language not supported:", lang)
        else: 
            print(file_in)
            result = annotate("json", 
                            dataset_type, 
                            lang, 
                            architecture=architecture, 
                            use_ELMo=use_ELMo, 
                            use_BERT=use_BERT,
                            file_in=file_in, 
                            file_out=file_out)
            """if result is not None:
                if file_out is None:
                    print(json.dumps(result, sort_keys=False, indent=4, ensure_ascii=False))
            """

    try:
        # see https://github.com/tensorflow/tensorflow/issues/3388
        K.clear_session()
    except:
        # TF could complain in some case
        print("\nLeaving TensorFlow...")
