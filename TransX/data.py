def load_data(*filelist:str) -> list:
    corpus_sum = []
    for filename in filelist:
        with open(filename,'r') as f:
            corpus = []
            for line in f:
                corpus.append(line.strip().split('\t'))
        corpus_sum.append(corpus)
    return corpus_sum

def load_entity_relation(filename:str) -> list:
    entity_list = []
    with open(filename,'r') as f:
        for line in f:
            entity_list.append(line.split('\t')[0])
    return entity_list