import os.path
import gzip
import tqdm


def read_gzfile(dir_path):
    '''
    dir_path: 压缩文件夹路径
    '''
    file_paths = [name for name in os.listdir('dir_path')
                  if os.path.isfile(os.path.join('dir_path', name))]
    for file_path in file_paths:
        # 以文本形式读取压缩文件
        with gzip.open(file_path, 'rt') as f:
            text = f.read()


def readFreebase(chunkNum, chunkSize=1024 * 1024):
    with gzip.open("data/freebase-rdf-latest.gz", 'rb') as fin, open("data/freebaseFirstClean.txt", 'w',
                                                                     encoding='utf-8') as fout:
        for i in tqdm.tqdm(range(chunkNum)):
            text = fin.readlines(chunkSize)
            if not text:
                print('finish!!!!')
                break
            for j in text:
                lineList = j.decode('utf-8').split('\t')
                for i in range(3):
                    index = lineList[i].find('http://rdf.freebase.com/ns/')
                    if index != -1:
                        lineList[i] = lineList[i][27:]
                index = lineList[1].find('#')
                if index != -1:
                    lineList[1] = lineList[1][index:]
                line = '\t'.join(lineList)
                fout.write(line)


def writeTemp(text):
    with open("data/freebaseFirstClean.txt", 'wb') as f:
        for i in text:
            f.write(i)


def getLineNum(filename='data/freebaseFirstClean.txt'):
    count = 0
    with open(filename, 'r', encoding='utf-8') as fin:
        while (True):
            text = fin.readline()
            if not text:
                break
            count += 1
    return count


def getFreebaseNumLine():
    writeBuffer = []
    with open("data/freebaseFirstClean.txt", 'r', encoding='utf-8') as fin, open("data/freebaseNumLine", 'w',
                                                                                 encoding='utf-8') as fout:
        for i in tqdm.tqdm(range(3130754000)):
            line = fin.readline()
            if not line:
                break
            lineList = line.split('\t')
            if lineList[2][0] != '"':
                continue
            if lineList[1][:-1].split('.')[-1] in ['display_name', 'text', 'key']:
                continue
            if not any(chr.isdigit() for chr in lineList[2]):
                continue
            writeBuffer.append(line)
            if i % 10000 == 0:
                for j in writeBuffer:
                    fout.write(j)
                writeBuffer.clear()
        for i in writeBuffer:
            fout.write(i)


def judgeListIsNum(str=''):
    str = str[1:-1]
    # print(str)
    if '^^' in str:
        return True
    strList = str.split('.')
    if len(strList) > 2:
        return False
    for i in strList:
        if not i.isdigit():
            return False
    return True


def freebaseNumFirstClean():
    with open('data/freebaseNumLine', 'r', encoding='utf-8') as fin, open('data/freebaseNumFirstClean.txt', 'w',
                                                                          encoding='utf-8') as fout:
        for i in tqdm.tqdm(range(268451900)):
            line = fin.readline()
            if not line:
                break
            lineList = line.split('\t')
            if judgeListIsNum(lineList[2]):
                fout.write(line)


def freebaseNumSecondClean():
    with open('data/freebaseNumFirstClean.txt', 'r', encoding='utf-8') as fin, open('data/freebaseNumSecondClean.txt',
                                                                                    'w', encoding='utf-8') as fout:
        for i in tqdm.tqdm(range(132390300)):
            line = fin.readline()
            if not line:
                break
            lineList = line.split('\t')
            if '^^' in lineList[2]:
                tempList = lineList[2].split('^^')
                if tempList[0][-1] != '"':
                    continue
                tempList[1] = tempList[1][tempList[1].index('#'):]
                lineList[2] = ''.join(tempList)
                line = '\t'.join(lineList)
            if 'key' in lineList[1]:
                lineList[1] = lineList[1][lineList[1].index('key'):]
                line = '\t'.join(lineList)
            fout.write(line)


def freebaseGetRelationList():
    RelationList = {}
    with open('data/freebaseNumSecondClean.txt', 'r', encoding='utf-8') as fin:
        for i in tqdm.tqdm(range(132390250)):
            line = fin.readline()
            if not line:
                break
            lineList = line.split('\t')
            if lineList[1] not in RelationList:
                RelationList[lineList[1]] = 1
            else:
                RelationList[lineList[1]] += 1
    RelationList = sorted(RelationList.items(), key=lambda x: x[1], reverse=True)
    with open('data/freebaseRelationList.txt', 'w', encoding='utf-8') as fout:
        for i in RelationList:
            fout.write(i[0] + '\t' + str(i[1]) + '\n')


def freebaseNumThirdClean():
    relation = []
    with open('data/freebaseRelationList.txt', 'r', encoding='utf-8') as relationfin, \
            open('data/freebaseNumSecondClean.txt', 'r', encoding='utf-8') as linefin, \
            open('data/freebaseNumThirdClean.txt', 'w', encoding='utf-8') as fout:
        for i in tqdm.tqdm(range(762)):
            line = relationfin.readline()
            if not line:
                break
            relation.append(line.split('\t')[0])
        for i in tqdm.tqdm(range(132390250)):
            line = linefin.readline()
            if not line:
                break
            lineRelation = line.split('\t')[1]
            if lineRelation in relation:
                fout.write(line)


def count():
    count = 0
    prefix = []
    with open("data/freebaseRelationList.txt", 'r', encoding='utf-8') as fin:
        for i in tqdm.tqdm(range(762)):
            line = fin.readline()
            if not line:
                break
            aprefix = line.split("\t")[0].split('.')[:-1]
            if aprefix not in prefix:
                prefix.append(aprefix)
                count += 1
    print(count)


def freebaseTripleCatch():
    writeBuffer = []
    with open("data/freebaseFirstClean.txt", 'r', encoding='utf-8') as fin, open("data/freebaseTripleFirstCatch.txt",
                                                                                 'w', encoding='utf-8') as fout:
        for i in tqdm.tqdm(range(3130754000)):
            line = fin.readline()
            if not line:
                break
            lineList = line.split('\t')
            if lineList[2][0] == '"':
                continue
            writeBuffer.append(line)
            if i % 10000 == 0:
                for j in writeBuffer:
                    fout.write(j)
                writeBuffer.clear()
        for i in writeBuffer:
            fout.write(i)


def freebaseTriplePrune():
    writeBuffer = []
    with open("data/freebaseTripleFirstCatch.txt", 'r', encoding='utf-8') as fin, \
            open("data/freebaseTripleSecondCatch.txt", 'w', encoding='utf-8') as fout:
        for i in tqdm.tqdm(range(1342085000)):
            line = fin.readline()
            if not line:
                break
            lineList = line.split('\t')
            if lineList[0][2] != '.' or lineList[2][2] != '.':
                continue
            writeBuffer.append(line)
            if i % 10000 == 0:
                for j in writeBuffer:
                    fout.write(j)
                writeBuffer.clear()
        for i in writeBuffer:
            fout.write(i)

def freebaseDateLine():
    writeBuffer = []
    with open("data/freebaseNumThirdClean.txt", 'r', encoding='utf-8') as fin, open("data/freebaseDateLine.txt", 'w',
                                                                                 encoding='utf-8') as fout:
        for i in tqdm.tqdm(range(132095600)):
            line = fin.readline()
            if not line:
                break
            lineList = line.split('\t')
            if '#' not in lineList[2]:
                continue
            writeBuffer.append(line)
            if i % 10000 == 0:
                for j in writeBuffer:
                    fout.write(j)
                writeBuffer.clear()
        for i in writeBuffer:
            fout.write(i)

def freebaseEntityList():
    pass

def freebaseTripleOnlyEntityCleaned():
    writeBuffer1 = []
    writebuffer2=[]
    entityList=set()
    flaghead=False
    flagtail=False
    with open("data/freebaseDateLine.txt", 'r', encoding='utf-8') as entityInFile,\
            open("data/freebaseTripleSecondCatch.txt",'r',encoding='utf-8')as tripleInFile,\
            open("data/freebaseEntityList.txt",'w',encoding='utf-8')as entityOutFile,\
            open("data/freebaseTripleOnlyCleanedOneIn.txt", 'w',encoding='utf-8') as fout1,\
            open("data/freebaseTripleOnlyCleanedAllIn.txt",'w',encoding='utf-8')as fout2:
        for i in tqdm.tqdm(range(17135200)):
            line = entityInFile.readline()
            if not line:
                break
            lineList = line.split('\t')
            entityList.add(lineList[0])
        for i in entityList:
            entityOutFile.write(i+'\n')
        for i in tqdm.tqdm(range(442157200)):
            line=tripleInFile.readline()
            if not line:
                break
            lineList=line.split('\t')
            if lineList[0] in entityList:
                flaghead=True
            else:
                flaghead=False
            if lineList[2] in entityList:
                flagtail=True
            else:
                flagtail=False
            if flaghead and flagtail:
                writebuffer2.append(line)
            if flaghead or flagtail:
                writeBuffer1.append(line)
            if i % 10000 == 0:
                for j in writeBuffer1:
                    fout1.write(j)
                for j in writebuffer2:
                    fout2.write(j)
                writeBuffer1.clear()
                writebuffer2.clear()
        for i in writeBuffer1:
            fout1.write(i)
        for i in writebuffer2:
            fout2.write(i)

def freebaseEntity2Date():
    entity2date={}
    with open("data/freebaseDateLine.txt",'r',encoding='utf-8')as fin:
        for i in tqdm.tqdm(range(17135200)):
            line=fin.readline()
            if not line:
                break
            entity=line.split('\t')[0]
            if entity not in entity2date:
                entity2date[entity]=[line[:-2]]
            else:
                entity2date[entity].append(line[:-2])
    return entity2date

def freebaseBuildTripleWithDate():
    entity2date=freebaseEntity2Date()
    with open("data/freebaseTripleOnlyCleanedAllIn.txt",'r',encoding='utf-8')as fin,open("data/freebaseTripleWithData.txt",'w',encoding='utf-8')as fout:
        for i in tqdm.tqdm(range(28127000)):
            line=fin.readline()
            if not line:
                break
            linelist=line.split('\t')
            for i in entity2date[linelist[0]]:
                for j in entity2date[linelist[2]]:
                    fout.write(i+linelist[1]+'\t'+j+'\n')

def freebaseBuildTripleForTrans():
    writebuffer=[]
    with open("data/freebaseTripleWithData.txt",'r',encoding='utf-8')as fin,open("data/freebaseTripleForTrans.txt",'w',encoding='utf-8')as fout:
        for i in tqdm.tqdm(range(38196800)):
            line=fin.readline()
            if not line:
                break
            lineList=line.split('\t')
            if lineList[2][1]=='T' or lineList[6][1]=='T':
                continue
            writebuffer.append(lineList[1]+'\t'+lineList[2][1:5]+'\t'+ lineList[3]+'\t'+ lineList[5]+'\t'+ lineList[6][1:5]+'\n')
            if i%1000==0:
                for j in writebuffer:
                    fout.write(j)
                writebuffer.clear()
        for j in writebuffer:
            fout.write(j)

# readFreebase(4*1024,100*1024*1024)
# print(getLineNum(filename='data/freebaseFirstClean.txt')) #3130753066
# getFreebaseNumLine()
# print(getLineNum(filename='data/freebaseNumLine')) #268451895
# freebaseNumFirstClean()
# print(getLineNum(filename='data/freebaseNumFirstClean.txt')) #132390248
# freebaseNumSecondClean()
# print(getLineNum(filename='data/freebaseNumSecondClean.txt')) #132390224
# freebaseGetRelationList() #762
# freebaseNumThirdClean()
# print(getLineNum(filename='data/freebaseNumThirdClean.txt')) #132095512
# count()
# freebaseTripleCatch()
# print(getLineNum(filename='data/freebaseTripleFirstCatch.txt')) #1342084643
# freebaseTriplePrune()
# print(getLineNum(filename='data/freebaseTripleSecondCatch.txt')) #442157143
# freebaseDateLine()
# print(getLineNum(filename='data/freebaseDateLine.txt')) #17135191
# freebaseTripleOnlyEntityCleaned()
# print(getLineNum(filename='data/freebaseEntityList.txt')) #15196913
# print(getLineNum(filename='data/freebaseTripleOnlyCleanedOneIn.txt')) #174368286
# print(getLineNum(filename='data/freebaseTripleOnlyCleanedAllIn.txt')) #28126956
# freebaseBuildTripleWithDate()
# print(getLineNum(filename="data/freebaseTripleWithData.txt")) #38196702
# freebaseBuildTripleForTrans()
# print(getLineNum(filename='data/freebaseTripleForTrans.txt')) #37927240