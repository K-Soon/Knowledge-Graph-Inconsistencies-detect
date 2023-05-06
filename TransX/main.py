from pickle import TRUE

from numpy.random.mtrand import hypergeometric
from data import load_data, load_entity_relation
from transE import transE
from transH import transH
import argparse


def main():
    parser = argparse.ArgumentParser(
        description='Train and test in TransH model')
    parser.add_argument('eproach', type=int, default=2000)
    parser.add_argument('batch', type=int, default=300)
    parser.add_argument('dimension', type=int, default=200)
    parser.add_argument('-L', help='Load the trained model',
                        action='store_true', default=False)
    parser.add_argument('-S', help='Save the trained model',
                        action='store_true', default=False)
    parser.add_argument('-lr', type=float,
                        help='learning rate', default=0.01)
    parser.add_argument('-mg', type=float, help='margin', default=1)
    parser.add_argument('-H', help='use transH, default = transE',
                        action='store_true', default=False)
    parser.add_argument('-E', help='emit predict for testcases',
                        action='store_true', default=False)
    parser.add_argument(
        '-T', help='evaluate model by hit score', type=int, default=0)
    args = parser.parse_args()
    train, test, dev = load_data(
        'data/train.txt', 'data/test.txt', 'data/dev.txt')  # only train is used
    if args.L:
        # these parameters can be left empty when filter = off
        if args.H:
            trans = transH(None, None, train)
        else:
            trans = transE(None, None, train)
        trans.load("data/trained_model")  # load model from file
        if args.E:
            trans.emit_predict(test, 'test_output.txt')
    else:
        train, test, dev = load_data(
            'data/train.txt', 'data/test.txt', 'data/dev.txt')  # only train is used
        entities = load_entity_relation('data/entity_with_text.txt')
        relations = load_entity_relation('data/relation_with_text.txt')
        if args.H:
            trans = transH(entities, relations, train,
                           args.dimension, args.lr, args.mg)
        else:
            trans = transE(entities, relations, train,
                           args.dimension, args.lr, args.mg)
        trans.emb_init()
        trans.train(args.eproach, args.batch)
        if args.S:
            trans.save("data/trained_model")
        if args.E:
            trans.emit_predict(test, 'test_output.txt')
    if args.T:
        # use dev data to test hit@n value
        print(trans.hit(dev, n=args.T, filter=True))


if __name__ == "__main__":
    main()
