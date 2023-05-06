# TransE/H implementation

This repo is submitted as experiment homework for [Web Info 2021](http://staff.ustc.edu.cn/~tongxu/webinfo/), and cannot be applied to other use directly (because TA've already done the eneity/relation - id replacement work, and the test dataset have no intersection with the one for training, which results in higher hit@10 score), instead, more of kind of reference. However, if you want, you can check instructions below.

## TransE

TransE part rewrites [Anery/transE](https://github.com/Anery/transE), bringing remarkable increasement in performance. Current training speed is 0.09s/400 eproch(batch size=1). Hit@10 for reference is 45.38 when dimension = 170 and eproch = 700000. (The commit message's hit@10=28% is calculated when I don't know there's no intersection)

Feel free to use for reference, I think the inner process is quite clear comparing with those using heavy framework like pytorch.

## TransH

TransE part rewrites [zqhead/TransH](https://github.com/zqhead/TransH), bringing remarkable increasement in performance.(I dunno why they use so much `deepcopy`) Current training speed is 0.42s/2000 (eproch * batch size). Hit@5 for reference is 15 when dimension = 170 and eproch = 2000, batch size = 300. There maybe some errors when implementing this algorithm, resulting in such a low score.

The default entry in `main.py` is transE, however, to run transH just replace all 'TransE' by 'TransH', their API are the same.

Because of potential failure in implementing, refer to this at your own risk.

## Run on dataset

If you want to run the code over dataset like FB15k, data preprocessing is required. Entity/relation name should be replaced by an unique id as below:

```
1178	136	11264
4336	158	5741
7503	169	3645
2017	121	9624
9379	158	4073
```

Some codes also need tuning, for example, `data/entity_with_text` is not quite used in training, it's safe to comment out those part and tune the code accordingly. And the `emb_init` part also need to be modified.