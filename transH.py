import math
import pickle
import time
import random
import numpy as np
from numpy.random.mtrand import rand
import tqdm


class transH():
    def __init__(self, triple_rel: list, dim: int = 100, lr: float = 0.01, margin: float = 1) -> None:
        self.entities = {}
        self.relations = {}
        self.hyper_relations = {}
        # self.triple_rels = [  entity,date,relation,entity,date   ]
        self.triple_rels = triple_rel
        self.updateEntitiesCount={}
        self.updateHyperRelationCount={}
        self.updateNormRelation={}
        self.dim = dim
        self.lr = lr
        self.margin = margin
        self.loss = 0
        self.right=0
        self.BGT = np.ones(dim)
        self.EQ = np.zeros(dim)
        self.LET = self.EQ - self.BGT
        print("transH object initalized")
        np.seterr(all='raise')

    def emb_init(self) -> None:
        ceil = 6 / math.sqrt(self.dim)
        relations = []
        entities = []
        for triple_rel in self.triple_rels:
            if triple_rel[0] not in entities:
                entities.append(triple_rel[0])
            if triple_rel[2] not in relations:
                relations.append(triple_rel[2])
            if triple_rel[3] not in entities:
                entities.append(triple_rel[3])
        self.relations = {relation: transH.norm(
            np.random.uniform(-ceil, ceil, self.dim)) for relation in relations}
        self.hyper_relations = {relation: transH.norm(
            np.random.uniform(-ceil, ceil, self.dim)) for relation in self.relations.keys()}
        self.entities = {entity: transH.norm(
            np.random.uniform(-ceil, ceil, self.dim)) for entity in entities}
        self.updateEntitiesCount={entity:0 for entity in entities}
        self.updateNormRelation={relation:0 for relation in relations}
        self.updateHyperRelationCount={relation:0 for relation in relations}
        self.BGT = self.norm(self.BGT)
        #self.EQ = self.norm(self.EQ)
        self.LET = self.norm(self.LET)
        print("transH embedding initalizd")

    def dist(self, h: np.array,wr:np.array ,dr: np.array, t: np.array,target) -> float:
        return np.sum(np.square(h - np.dot(wr, h) * wr + dr - t + np.dot(wr, t) * wr-target+np.dot(wr,target)*wr))




    @staticmethod
    def norm(vector: np.array) -> np.array:
        return vector / np.linalg.norm(vector, ord=2)
    '''
    def corrupt(self, head, tail, tph):
        tph = tph * tph
        weight = tph / (tph + 1)
        if random.random() < weight:  # tail per head possibly greater than 1, tend to replace head entity
            fake_head = head
            while fake_head == head:  # prevent from sampling the right one
                fake_head = random.sample(self.entities.keys(), 1)[0]
            return fake_head, tail
        else:
            fake_tail = tail
            while fake_tail == tail:
                fake_tail = random.sample(self.entities.keys(), 1)[0]
            return head, fake_tail
        '''

    def train(self, eprochs: int, batch: int, arrayMaxValue=2,arrayValueInfluence=1,w_d_influence=0.5,w_d_margin=0.05) -> None:
        # here we use Stochastic Gradient Descent.
        print(f"transH training, batch size: {batch}, eproch: {eprochs}")
        interval = 2000 // batch
        start_timer = time.time()
        for epoch in range(1,eprochs+1):
            if epoch % interval == 0:
                print("eproch: %d\t loss: %.1f\t right: %.1f\t acc: %.3f\t time: %.2fs" %
                      (epoch, self.loss,self.right,self.right/(self.loss+self.right) ,time.time() - start_timer))
                start_timer = time.time()
                self.loss = 0
                self.right=0
            rel_batch = random.sample(self.triple_rels, batch)
            self.update_embedding(rel_batch,arrayMaxValue,arrayValueInfluence,w_d_influence,w_d_margin)

    def update_embedding(self, rel_batchs,arrayMaxValue,arrayValueInfluence,w_d_influence,w_d_margin) -> None:
        # sometimes the random sample above will return list with 5 elements (should be 3)
        # known bug
        batch_entities = {}  # eneities to update
        batch_relations = {}  # relations to update
        batch_hyper_relations = {}
        for rel_batch in rel_batchs:
            rel_head, relation, rel_tail = rel_batch[0], rel_batch[2], rel_batch[3]
            if rel_batch[1]>rel_batch[4]:
                target=self.BGT
            elif rel_batch[1]<rel_batch[4]:
                target=self.LET
            else:
                target=self.EQ
            try:
                rel_dist = self.dist(self.entities[rel_head],self.relations[relation], self.hyper_relations[relation], self.entities[rel_tail],
                                 target)
                corr_dist = self.dist(self.entities[rel_head],self.relations[relation], self.hyper_relations[relation], self.entities[rel_tail],
                                  -1*target)
            except:
                print(rel_batch)
                print(self.updateEntitiesCount[rel_head],self.updateNormRelation[relation],self.updateHyperRelationCount[relation],self.updateEntitiesCount[rel_tail])
                print(self.entities[rel_head])
                print(self.relations[relation])
                print(self.hyper_relations[relation])
                print(self.entities[rel_tail])
                exit(0)
            # hinge loss
            loss = rel_dist - corr_dist + self.margin
            if loss >= 0:
                self.loss += 1
                if rel_head not in batch_entities:
                    # this will perform a copy indeed
                    batch_entities[rel_head] = self.entities[rel_head]
                if rel_tail not in batch_entities:
                    batch_entities[rel_tail] = self.entities[rel_tail]
                if relation not in batch_relations:
                    batch_relations[relation] = self.relations[relation]
                if relation not in batch_hyper_relations:
                    batch_hyper_relations[relation] = self.hyper_relations[relation]
                grad_norm=2*(self.entities[rel_head]-np.dot(self.relations[relation],self.entities[rel_head])*self.relations[relation]+
                             self.hyper_relations[relation]-
                             self.entities[rel_tail]+np.dot(self.relations[relation],self.entities[rel_tail])*self.relations[relation]-
                             target+np.dot(self.relations[relation],target)*self.relations[relation])
                grad_head=grad_norm*(1-self.relations[relation]**2)
                grad_tail=-grad_head
                grad_hyper=grad_norm*2*(self.entities[rel_tail]+target-self.entities[rel_head])*self.relations[relation]

                '''                
                if np.linalg.norm(batch_entities[rel_head])>arrayMaxValue:
                    grad_head+=2*self.entities[rel_head]*arrayValueInfluence
                if np.linalg.norm(batch_entities[rel_tail])>arrayMaxValue:
                    grad_tail+=2*self.entities[rel_tail]*arrayValueInfluence
                if np.linalg.norm(batch_relations[relation])>arrayMaxValue:
                    grad_norm+=2*self.relations[relation]*arrayValueInfluence
                '''

                if abs(np.dot(self.relations[relation],self.hyper_relations[relation])/np.linalg.norm(self.hyper_relations[relation]))>w_d_margin:
                    grad_norm+=2*w_d_influence*np.dot(self.relations[relation],self.hyper_relations[relation])*self.hyper_relations[relation]/(np.linalg.norm(self.hyper_relations[relation])**2)

                grad_head*=self.lr
                grad_tail*=self.lr
                grad_hyper*=self.lr
                grad_norm*=self.lr

                # update
                if self.updateEntitiesCount[rel_head]==0:
                    batch_entities[rel_head]-=grad_head
                    self.updateEntitiesCount[rel_head]+=1
                elif self.updateEntitiesCount[rel_tail]==0:
                    batch_entities[rel_tail]-=grad_tail
                    self.updateEntitiesCount[rel_tail]+=1
                elif self.updateNormRelation[relation]==0:
                    batch_relations[relation]-=grad_norm
                    self.updateNormRelation[relation]+=1
                elif self.updateHyperRelationCount[relation]==0:
                    batch_hyper_relations[relation]-=grad_hyper
                    self.updateHyperRelationCount[relation]+=1
                else:
                    a,b,c=1/self.updateEntitiesCount[rel_head],1/self.updateHyperRelationCount[relation],1/self.updateEntitiesCount[rel_tail]
                    sum=a+b+c
                    a,b,c=a/sum,b/sum,c/sum
                    batch_entities[rel_head] -= grad_head *a
                    batch_relations[relation] -= grad_norm *b
                    batch_hyper_relations[relation] -= grad_hyper * b
                    batch_entities[rel_tail] -= grad_tail * c
                    self.updateEntitiesCount[rel_head] += 1
                    self.updateEntitiesCount[rel_tail] += 1
                    self.updateHyperRelationCount[relation] += 1
                    self.updateNormRelation[relation] += 1

                # wr**2=1
                batch_relations[relation]=self.norm(batch_relations[relation])

                if np.linalg.norm(batch_entities[rel_head])>arrayMaxValue:
                    batch_entities[rel_head]=self.norm(batch_entities[rel_head])
                if np.linalg.norm(batch_entities[rel_tail])>arrayMaxValue:
                    batch_entities[rel_tail]=self.norm(batch_entities[rel_tail])
                if np.linalg.norm(batch_relations[relation])>arrayMaxValue:
                    batch_relations[relation]=self.norm(batch_relations[relation])
                if np.linalg.norm(batch_hyper_relations[relation])>arrayMaxValue:
                    batch_hyper_relations[relation]=self.norm(batch_hyper_relations[relation])
            else:
                self.right+=1

        for entity in batch_entities.keys():
            self.entities[entity] = batch_entities[entity]
        for relation in batch_relations.keys():
            self.relations[relation] = batch_relations[relation]
        for relation in batch_hyper_relations.keys():
            self.hyper_relations[relation] = batch_hyper_relations[relation]

    def save(self, filename):
        data = [self.entities, self.relations]
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"Model saved to {filename}")

    def load(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        self.entities, self.relations = data
        print(f"Model loaded from {filename}")


    def judge(self):
        correct=0
        error=0
        equal=0
        for rel_triple in self.triple_rels:
            flagBGT=self.dist(self.entities[rel_triple[0]],self.relations[rel_triple[2]],self.hyper_relations[rel_triple[2]],self.entities[rel_triple[3]],self.BGT)
            flagLET=self.dist(self.entities[rel_triple[0]],self.relations[rel_triple[2]],self.hyper_relations[rel_triple[2]],self.entities[rel_triple[3]],self.LET)

            if rel_triple[1] > rel_triple[4]:
                if flagBGT<flagLET:
                    correct+=1
                else:
                    error+=1
            elif rel_triple[1] < rel_triple[4]:
                if flagBGT>flagLET:
                    correct+=1
                else:
                    error+=1
            else:
                equal+=1
        print(correct,error,correct/(correct+error),equal)

'''
    # the classic way is pretty slow due to enormous distance calculations
    def hit(self, testdata, n: int = 10, filter=False, ceiling: int = 500) -> float:
        assert not filter or self.contain
        hit = 0
        count = 1
        for head, rel, tail in random.sample(testdata, ceiling):
            if count % 20 == 0:
                print("%d/%d tested\t hit@%d rate: %.2f" %
                      (count, len(testdata), n, hit/count))
            result = {}
            try:
                for entity in self.entities.keys():
                    # in this dataset, the triple in train will not occur in test/dev
                    # comment out `if` below and `self.contain` in __init__ if this cannot be satisfied
                    if filter and (head, rel, entity) in self.contain:
                        continue
                    result[self.test_distance_score(head, rel, entity)] = entity
                result = dict(sorted(result.items())[:n])
            except: # head / relation not occurred in training dataset
                result = random.sample(self.entities.keys(), 5)
            if tail in result.values():
                hit += 1
            count += 1
        hit /= len(testdata)
        return hit

    def emit_predict(self, testdata, savefile: str) -> None:
        count = 1
        with open(savefile, 'w') as f:
            for head, rel, _ in testdata:
                if count % 20 == 0:
                    print(f'{count}/{len(testdata)} emitted')
                result = {}
                try:
                    for entity in self.entities.keys():
                        if (head, rel, entity) in self.contain:
                            continue
                        result[self.test_distance_score(
                            head, rel, entity)] = entity
                    result = dict(sorted(result.items())[:5]).values()
                except:  # head / relation not occurred in training dataset
                    result = random.sample(self.entities.keys(), 5)
                f.write(','.join(result)+'\n')
                count += 1
'''


def getData(filename="data/freebaseTripleForTrans.txt",linenum=37927300):
    linenum=int(linenum/20)
    triples = []
    with open(filename, 'r', encoding='utf-8') as fin:
        for i in tqdm.tqdm(range(linenum)):
            line = fin.readline()
            if not line:
                break
            lineList = line.split('\t')

            lineList[1]=int(lineList[1])
            lineList[4]=int(lineList[4])

            triples.append(lineList)

    print("loadData Done!")
    return triples
def saveTriple(triples):
    with open("data/tripleless",'wb')as f:
        pickle.dump(triples,f)
    print("Triple Saved!")

def loadTriple():
    with open("data/tripleless",'rb')as f:
        triples=pickle.load(f)
    print("Triple Loaded!")
    return triples

#saveTriple(getData())
triples=loadTriple()
trans=transH(triple_rel=triples,dim=150,lr=0.2,margin=0)
trans.emb_init()
trans.train(eprochs=5000,batch=300)
trans.judge()

