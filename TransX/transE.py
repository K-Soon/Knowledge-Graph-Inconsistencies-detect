import math
import pickle
from os import stat
import time
import random
import numpy as np

from data import load_entity_relation


class transE():
    def __init__(self, entity: list, relation: list, triple_rel: list, dim: int = 100, lr: float = 0.01, margin: float = 1) -> None:
        #self.entities = entity
        #self.relations = relation
        self.entities = {}
        self.relations = {}
        self.triple_rels = triple_rel
        self.dim = dim
        self.lr = lr
        self.margin = margin
        self.loss = 0
        self.contain = {(head, rel, tail) for head, rel, tail in triple_rel}
        print("transE object initalized")

    def emb_init(self) -> None:
        ceil = 6/math.sqrt(self.dim)
        self.relations = {relation: transE.norm(
            np.random.uniform(-ceil, ceil, self.dim)) for relation in self.relations}
        self.entities = {entity: transE.norm(
            np.random.uniform(-ceil, ceil, self.dim)) for entity in self.entities}
        # because there are entities/relations not given in entity_with_text nor relation_with_text
        for triple_rel in self.triple_rels:
            if triple_rel[0] not in self.entities:
                self.entities[triple_rel[0]] = transE.norm(
                    np.random.uniform(-ceil, ceil, self.dim))
            if triple_rel[1] not in self.relations:
                self.relations[triple_rel[1]] = transE.norm(
                    np.random.uniform(-ceil, ceil, self.dim))
            if triple_rel[2] not in self.entities:
                self.entities[triple_rel[2]] = transE.norm(
                    np.random.uniform(-ceil, ceil, self.dim))
        print("transE embedding initalizd")

    @staticmethod
    def dist_l1(h: np.array, r: np.array, t: np.array) -> float:
        return np.sum(np.fabs(h+r-t))

    @staticmethod
    def dist_l2(h: np.array, r: np.array, t: np.array) -> float:
        return np.sum(np.square(h+r-t))

    @staticmethod
    def norm(vector: np.array) -> np.array:
        return vector/np.linalg.norm(vector, ord=2)

    def corrupt(self, head, tail):
        if random.randint(0, 1):
            fake_head = head
            while fake_head == head:  # prevent from sampling the right one
                fake_head = random.sample(self.entities.keys(), 1)[0]
            return fake_head, tail
        else:
            fake_tail = tail
            while fake_tail == tail:
                fake_tail = random.sample(self.entities.keys(), 1)[0]
            return head, fake_tail

    def train(self, eprochs: int, batch: int) -> None:
        # here we use Stochastic Gradient Descent.
        print(f"transE training, batch size: {batch}, eproch: {eprochs}")
        interval = 2000//batch
        start_timer = time.time()
        for epoch in range(1, eprochs+1):
            if epoch % interval == 0:
                print("eproch: %d\t loss: %.2f\t time: %.2f" %
                      (epoch, self.loss, time.time()-start_timer))
                start_timer = time.time()
                self.loss = 0
            rel_batch = random.sample(self.triple_rels, batch)
            for rel_triple in rel_batch:
                rel_triple.extend(
                    list(self.corrupt(rel_triple[0], rel_triple[2])))
            self.update_embedding(rel_batch, self.dist_l2)

    def update_embedding(self, rel_batchs, dist=dist_l2) -> None:
        # sometimes the random sample above will return list with 5 elements (should be 3)
        # unknown bug
        batch_entities = {}
        batch_relations = {}
        for rel_batch in rel_batchs:
            rel_head, relation, rel_tail, corr_head, corr_tail = rel_batch[:5]
            rel_dist = dist(
                self.entities[rel_head], self.relations[relation], self.entities[rel_tail])
            corr_dist = dist(
                self.entities[corr_head], self.relations[relation], self.entities[corr_tail])
            # hinge loss
            loss = rel_dist-corr_dist+self.margin
            if loss > 0:
                self.loss += loss

                if rel_head not in batch_entities:
                    # this will perform a copy indeed
                    batch_entities[rel_head] = self.entities[rel_head]
                if rel_tail not in batch_entities:
                    batch_entities[rel_tail] = self.entities[rel_tail]
                if relation not in batch_relations:
                    batch_relations[relation] = self.relations[relation]
                if corr_head not in batch_entities:
                    # this will perform a copy indeed
                    batch_entities[corr_head] = self.entities[corr_head]
                if corr_tail not in batch_entities:
                    # this will perform a copy indeed
                    batch_entities[corr_tail] = self.entities[corr_tail]

                grad_pos = 2 * \
                    (self.entities[rel_head] +
                     self.relations[relation]-self.entities[rel_tail])
                grad_neg = 2 * \
                    (self.entities[corr_head] +
                     self.relations[relation]-self.entities[corr_tail])

                # update
                grad_pos *= self.lr
                batch_entities[rel_head] -= grad_pos
                batch_entities[rel_tail] += grad_pos

                # head entity replaced
                grad_neg *= self.lr
                if corr_head == rel_head:  # move away from wrong relationships
                    batch_entities[rel_head] += grad_neg
                    batch_entities[corr_tail] -= grad_neg
                # tail entity replaced
                else:
                    batch_entities[corr_head] += grad_neg
                    batch_entities[rel_tail] -= grad_neg

                # relation update
                batch_relations[relation] -= grad_pos
                batch_relations[relation] += grad_neg

        for entity in batch_entities.keys():
            self.entities[entity] = self.norm(batch_entities[entity])
        for relation in batch_relations.keys():
            self.relations[relation] = self.norm(batch_relations[relation])

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

    # the classic way is pretty slow due to enormous distance calculations
    def hit(self, testdata, n: int = 10, filter=False, ceiling: int = 5000) -> float:
        assert not filter or self.contain
        hit = 0
        count = 1
        for head, rel, tail in random.sample(testdata, ceiling):
            if count % 20 == 0:
                print("%d/%d tested\t hit@%d rate: %.2f" %
                      (count, len(testdata), n, hit/count))
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
                except: # head / relation not in training dataset
                    result = random.sample(self.entities.keys(), 5)
                f.write(','.join(result)+'\n')
                count += 1
