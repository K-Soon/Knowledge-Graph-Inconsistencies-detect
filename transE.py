import math
import pickle
import tqdm
import time
import random
import numpy as np
import gc


class transE():
    def __init__(self, triple_rel: list, dim: int = 100, lr: float = 0.01,
                 margin: float = 1) -> None:
        # self.entities = entity
        # self.relations = relation

        # self.triples = [  entity,date,relation,entity,date   ]

        self.entities = {}
        self.relations = {}
        self.updateCount = {}
        self.triples = triple_rel
        self.rel_triples = [i for i in self.triples]
        self.errorList=[]
        self.dim = dim
        self.lr = lr
        self.margin = margin
        self.loss = 0
        self.right = 0
        self.BGT = np.ones(dim)
        self.EQ = np.zeros(dim)
        self.LET = self.EQ - self.BGT
        print("transE object initialized")

    def error_insert(self, proportion: float):
        for i in range(len(self.triples)):
            if random.random() < proportion:
                self.triples[i][1], self.triples[i][4] = self.triples[i][4], self.triples[i][1]
                self.errorList.append(i)

    def emb_init(self) -> None:
        entities = set()
        relations = set()
        for triple_rel in self.triples:
            if triple_rel[0] not in entities:
                entities.add(triple_rel[0])
            if triple_rel[2] not in relations:
                relations.add(triple_rel[2])
            if triple_rel[3] not in entities:
                entities.add(triple_rel[3])

        ceil = 6 / math.sqrt(self.dim)
        for i in entities:
            if i not in self.updateCount:
                self.updateCount[i] = 0
        for i in relations:
            if i not in self.updateCount:
                self.updateCount[i] = 0
        self.relations = {relation: transE.norm(
            np.random.uniform(-ceil, ceil, self.dim)) for relation in relations}
        self.entities = {entity: transE.norm(
            np.random.uniform(-ceil, ceil, self.dim)) for entity in entities}
        self.BGT = self.norm(self.BGT)
        # self.EQ = self.norm(self.EQ)
        self.LET = self.norm(self.LET)
        # because there are entities/relations not given in entity_with_text nor relation_with_text
        print("transE embedding initialized")

    '''
    @staticmethod
    def dist_l1(h: np.array, r: np.array, t: np.array) -> float:
        return np.sum(np.fabs(h + r - t))
    '''

    def dist(self, h: np.array, r: np.array, t: np.array, target: np.array) -> float:
        return np.sum(np.square(h + r - t - target))

    @staticmethod
    def norm(vector: np.array) -> np.array:
        return vector / np.linalg.norm(vector, ord=2)

    def train(self, epochs: int, batch: int) -> None:
        # here we use Stochastic Gradient Descent.
        print(f"transE training, batch size: {batch}, epoch: {epochs}")
        interval = 2000 // batch
        start_timer = time.time()
        for epoch in range(1, epochs + 1):
            if epoch % interval == 0:
                print("eproch: %d\t loss: %.1f\t right: %.1f\t acc: %.3f\t time: %.2fs" %
                      (epoch, self.loss, self.right, self.right / (self.loss + self.right), time.time() - start_timer))
                start_timer = time.time()
                self.loss = 0
                self.right = 0
            rel_batch = random.sample(self.triples, batch)
            self.update_embedding(rel_batch)

    def update_embedding(self, rel_batches) -> None:
        # sometimes the random sample above will return list with 5 elements (should be 3)
        # unknown bug
        batch_entities = {}
        batch_relations = {}
        for rel_batch in rel_batches:
            rel_head, relation, rel_tail = rel_batch[0], rel_batch[2], rel_batch[3]
            flagEQ = 0
            if rel_batch[1] > rel_batch[4]:
                target = self.BGT
            elif rel_batch[1] < rel_batch[4]:
                target = self.LET
            else:
                target = self.BGT
                flagEQ = 1
            rel_dist = self.dist(self.entities[rel_head], self.relations[relation], self.entities[rel_tail],
                                 target)
            corr_dist = self.dist(self.entities[rel_head], self.relations[relation], self.entities[rel_tail],
                                  -1 * target)
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
                grad_pos = 2 * \
                           (self.entities[rel_head] +
                            self.relations[relation] - self.entities[rel_tail] - target)

                # update
                grad_pos *= self.lr
                if self.updateCount[rel_head] == 0:
                    batch_entities[rel_head] -= grad_pos
                    self.updateCount[rel_head] += 1
                elif self.updateCount[rel_tail] == 0:
                    batch_entities[rel_tail] += grad_pos
                    self.updateCount[rel_tail] += 1
                elif self.updateCount[relation] == 0:
                    batch_relations[relation] -= grad_pos
                    self.updateCount[relation] += 1
                else:
                    x, y, z = 1 / self.updateCount[rel_head], 1 / self.updateCount[relation], 1 / self.updateCount[
                        rel_tail]
                    sum = x + y + z
                    # 使刚开始的变化大，后来逐渐减小。86->93
                    x, y, z = x / sum, y / sum, z / sum
                    choice = random.random()
                    batch_entities[rel_head] -= grad_pos * x
                    self.updateCount[rel_head] += 1

                    batch_relations[relation] -= grad_pos * y
                    self.updateCount[relation] += 1

                    batch_entities[rel_tail] += grad_pos * z
                    self.updateCount[rel_tail] += 1

                batch_entities[rel_head] = self.norm(batch_entities[rel_head])
                batch_entities[rel_tail] = self.norm(batch_entities[rel_tail])
                batch_relations[relation] = self.norm(batch_relations[relation])

                # batch_entities[rel_head] -= grad_pos
                # batch_entities[rel_tail] += grad_pos
                '''
                # head entity replaced
                grad_neg *= self.lr
                if corr_head == rel_head:  # move away from wrong relationships
                    batch_entities[rel_head] += grad_neg
                    batch_entities[corr_tail] -= grad_neg
                # tail entity replaced
                else:
                    batch_entities[corr_head] += grad_neg
                    batch_entities[rel_tail] -= grad_neg
                '''
                # relation update
                # batch_relations[relation] -= grad_pos
                # batch_relations[relation] += grad_neg
            else:
                self.right += 1

        for entity in batch_entities.keys():
            self.entities[entity] = batch_entities[entity]
        for relation in batch_relations.keys():
            self.relations[relation] = batch_relations[relation]

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
        correct = 0
        error = 0
        equal = 0
        for rel_triple in self.rel_triples:
            flagBGT = self.dist(self.entities[rel_triple[0]], self.relations[rel_triple[2]],
                                self.entities[rel_triple[3]], self.BGT)
            flagLET = self.dist(self.entities[rel_triple[0]], self.relations[rel_triple[2]],
                                self.entities[rel_triple[3]], self.LET)
            if rel_triple[1] > rel_triple[4]:
                if flagBGT < flagLET:
                    correct += 1
                else:
                    error += 1
            elif rel_triple[1] < rel_triple[4]:
                if flagBGT > flagLET:
                    correct += 1
                else:
                    error += 1
            else:
                equal += 1
        print(correct, error, correct / (correct + error), equal)
        correct=0
        error=0
        for i in self.errorList:
            rel_triple=self.triples[i]
            flagBGT = self.dist(self.entities[rel_triple[0]], self.relations[rel_triple[2]],
                                self.entities[rel_triple[3]], self.BGT)
            flagLET = self.dist(self.entities[rel_triple[0]], self.relations[rel_triple[2]],
                                self.entities[rel_triple[3]], self.LET)
            if rel_triple[1] > rel_triple[4]:
                if flagBGT < flagLET:
                    error += 1
                else:
                    correct += 1
            elif rel_triple[1] < rel_triple[4]:
                if flagBGT > flagLET:
                    error += 1
                else:
                    correct += 1
            else:
                equal += 1
        print("Recall::" ,correct, error, correct / (correct + error), equal)


    '''
    # the classic way is pretty slow due to enormous distance calculations
    def hit(self, testdata, n: int = 10, filter=False, ceiling: int = 5000) -> float:
        assert not filter or self.contain
        hit = 0
        count = 1
        for head, rel, tail in random.sample(testdata, ceiling):
            if count % 20 == 0:
                print("%d/%d tested\t hit@%d rate: %.2f" %
                      (count, len(testdata), n, hit / count))
            assume_tail = self.entities[head] + self.relations[rel]
            result = {}
            for entity in self.entities.keys():
                # in this dataset, the triple in train will not occur in test/dev
                # comment out `if` below and `self.contain` in __init__ if this cannot be satisfied
                if filter and (head, rel, entity) in self.contain:
                    continue
                result[np.sum(
                    np.square(assume_tail - self.entities[entity]))] = entity
            result = dict(sorted(result.items())[:n])
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
                assume_tail = self.entities[head] + self.relations[rel]
                result = {}
                try:
                    for entity in self.entities.keys():
                        if (head, rel, entity) in self.contain:
                            continue
                        result[np.sum(
                            np.square(assume_tail - self.entities[entity]))] = entity
                    result = dict(sorted(result.items())[:5]).values()
                except:  # head / relation not in training dataset
                    result = random.sample(self.entities.keys(), 5)
                f.write(','.join(result) + '\n')
                count += 1
    '''


def getData(filename="data/freebaseTripleForTrans.txt", linenum=37927300):
    linenum = int(linenum / 20)
    triples = []
    with open(filename, 'r', encoding='utf-8') as fin:
        for i in tqdm.tqdm(range(linenum)):
            line = fin.readline()
            if not line:
                break
            lineList = line.split('\t')

            lineList[1] = int(lineList[1])
            lineList[4] = int(lineList[4])
            if lineList[1] == lineList[4]:
                continue
            triples.append(lineList)

    print("loadData Done!")
    return triples


def saveTriple(triples):
    with open("data/tripleless", 'wb') as f:
        pickle.dump(triples, f)
    print("Triple Saved!")


def loadTriple():
    with open("data/tripleless", 'rb') as f:
        triples = pickle.load(f)
    print("Triple Loaded!")
    return triples


# saveTriple(getData())
triples = loadTriple()
trans = transE(triple_rel=triples, dim=120, lr=0.3, margin=0)
# 0.1:0.71  0.01:0.93   0.02:0.90   0.03:0.85
trans.error_insert(0.01)
trans.emb_init()
trans.train(epochs=5000, batch=300)
trans.judge()
