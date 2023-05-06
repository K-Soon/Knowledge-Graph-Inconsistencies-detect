import math
import pickle
from os import stat
import time
import random
import numpy as np
from numpy.random.mtrand import rand

from data import load_entity_relation


class transH():
    def __init__(self, entity: list, relation: list, triple_rel: list, dim: int = 100, lr: float = 0.01, margin: float = 1) -> None:
        self.entities = entity
        self.relations = relation
        self.hyper_relations = {}
        self.tail_per_head = {}
        self.triple_rels = triple_rel
        self.contain = {(head, rel, tail) for head, rel, tail in triple_rel}
        self.dim = dim
        self.lr = lr
        self.margin = margin
        self.loss = 0
        print("transH object initalized")
        np.seterr(all='raise')

    def emb_init(self) -> None:
        ceil = 6/math.sqrt(self.dim)
        self.hyper_relations = {relation: transH.norm(
            np.random.uniform(-ceil, ceil, self.dim)) for relation in self.relations}
        self.relations = {relation: transH.norm(
            np.random.uniform(-ceil, ceil, self.dim)) for relation in self.relations}
        self.hyper_relations = {relation: transH.norm(
            np.random.uniform(-ceil, ceil, self.dim)) for relation in self.relations.keys()}
        self.entities = {entity: transH.norm(
            np.random.uniform(-ceil, ceil, self.dim)) for entity in self.entities}
        rel_eneities = {}
        for triple_rel in self.triple_rels:
            if triple_rel[0] not in self.entities:
                self.entities[triple_rel[0]] = transH.norm(
                    np.random.uniform(-ceil, ceil, self.dim))
            if triple_rel[1] not in self.relations:
                self.relations[triple_rel[1]] = transH.norm(
                    np.random.uniform(-ceil, ceil, self.dim))
            if triple_rel[1] not in self.hyper_relations:
                self.hyper_relations[triple_rel[1]] = transH.norm(
                    np.random.uniform(-ceil, ceil, self.dim))
            if triple_rel[2] not in self.entities:
                self.entities[triple_rel[2]] = transH.norm(
                    np.random.uniform(-ceil, ceil, self.dim))
            if triple_rel[1] not in self.tail_per_head:
                rel_eneities[triple_rel[1]] = (
                    set(triple_rel[0]), set(triple_rel[2]))
            else:
                rel_eneities[triple_rel[1]][0].add(triple_rel[0])
                rel_eneities[triple_rel[1]][1].add(triple_rel[2])
        for relation in rel_eneities:
            self.tail_per_head[relation] = len(rel_eneities[relation][
                1]) / len(rel_eneities[relation][0])
        print("transH embedding initalizd")

    @staticmethod
    def dist_l2(h: np.array, r: np.array, r_hyper: np.array, t: np.array) -> float:
        return np.sum(np.square(h+r_hyper-t+np.dot(r, t-h)*r))

    @staticmethod
    def norm(vector: np.array) -> np.array:
        return vector/np.linalg.norm(vector, ord=2)

    @staticmethod
    def project(a: np.array, b: np.array) -> np.array:
        return a - np.dot(a, b)*b

    def test_distance_score(self, head, relation, tail) -> float:
        head_hyper = transH.project(
            self.entities[head], self.relations[relation])
        tail_hyper = transH.project(
            self.entities[tail], self.relations[relation])
        return np.sum(np.square(head_hyper-tail_hyper+self.hyper_relations[relation]))

    def corrupt(self, head, tail, tph):
        tph = tph*tph
        weight = tph/(tph+1)
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

    def train(self, eprochs: int, batch: int) -> None:
        # here we use Stochastic Gradient Descent.
        print(f"transH training, batch size: {batch}, eproch: {eprochs}")
        interval = 2000//batch
        start_timer = time.time()
        for epoch in range(eprochs):
            if epoch % interval == 0:
                print("eproch: %d\t loss: %.2f\t time: %.2fs" %
                      (epoch, self.loss, time.time()-start_timer))
                start_timer = time.time()
                self.loss = 0
            rel_batch = random.sample(self.triple_rels, batch)
            for rel_triple in rel_batch:
                rel_triple.extend(
                    list(self.corrupt(rel_triple[0], rel_triple[2], self.tail_per_head[rel_triple[1]])))
            self.update_embedding(rel_batch,)

    def update_embedding(self, rel_batchs) -> None:
        # sometimes the random sample above will return list with 5 elements (should be 3)
        # known bug
        batch_entities = {}  # eneities to update
        batch_relations = {}  # relations to update
        batch_hyper_relations = {}
        for rel_batch in rel_batchs:
            rel_head, relation, rel_tail, corr_head, corr_tail = rel_batch[:5]
            rel_dist = transH.dist_l2(
                self.entities[rel_head], self.relations[relation], self.hyper_relations[relation], self.entities[rel_tail])
            corr_dist = transH.dist_l2(
                self.entities[corr_head], self.relations[relation], self.hyper_relations[relation], self.entities[corr_tail])
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
                if relation not in batch_hyper_relations:
                    batch_hyper_relations[relation] = self.hyper_relations[relation]
                if corr_head not in batch_entities:
                    # this will perform a copy indeed
                    batch_entities[corr_head] = self.entities[corr_head]
                if corr_tail not in batch_entities:
                    # this will perform a copy indeed
                    batch_entities[corr_tail] = self.entities[corr_tail]

                coeff = 1 - \
                    np.dot(self.relations[relation], self.relations[relation])
                grad_right = self.entities[rel_head]-self.entities[rel_tail] -\
                    self.relations[relation]*np.dot(
                    self.relations[relation], (self.entities[rel_head]-self.entities[rel_tail]))
                grad_wrong = self.entities[corr_head]-self.entities[corr_tail] -\
                    self.relations[relation]*np.dot(
                    self.relations[relation], (self.entities[corr_head]-self.entities[corr_tail]))
                grad_correct = (
                    grad_right + self.hyper_relations[relation]) * 2*coeff
                grad_corrupt = (
                    grad_wrong + self.hyper_relations[relation]) * 2*coeff
                grad_hyper = (grad_right-grad_wrong)*2
                grad_norm = 4*np.dot((grad_right+self.hyper_relations[relation]), (self.entities[rel_tail]-self.entities[rel_head]))*self.relations[relation] -\
                    4*np.dot((grad_wrong+self.hyper_relations[relation]), (
                        self.entities[corr_tail]-self.entities[corr_head]))*self.relations[relation]
                # update
                grad_correct *= self.lr
                batch_entities[rel_head] -= grad_correct * self.lr
                batch_entities[rel_tail] += grad_correct * self.lr

                # head entity replaced
                grad_corrupt *= self.lr
                if corr_head == rel_head:  # move away from wrong relationships
                    batch_entities[rel_head] += grad_corrupt * self.lr
                    batch_entities[corr_tail] -= grad_corrupt * self.lr
                # tail entity replaced
                else:
                    batch_entities[corr_head] += grad_corrupt * self.lr
                    batch_entities[rel_tail] -= grad_corrupt * self.lr

                # relation update
                batch_relations[relation] -= grad_norm * self.lr
                batch_hyper_relations[relation] -= grad_hyper * self.lr

        for entity in batch_entities.keys():
            self.entities[entity] = self.norm(batch_entities[entity])
        for relation in batch_relations.keys():
            self.relations[relation] = self.norm(batch_relations[relation])
        for relation in batch_hyper_relations.keys():
            # no norm here
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
