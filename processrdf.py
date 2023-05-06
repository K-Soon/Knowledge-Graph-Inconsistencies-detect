import rdflib



# 创建一个空的知识图谱对象
graph = rdflib.Graph()

# 从文件中读取NT数据
graph.parse("preprocessed.nt", format="n3")
print(graph)

def extract_numeric_data():
    numeric_data = {}
    for s, p, o in graph:
        if isinstance(o, rdflib.term.Literal) and o.datatype and o.datatype.startswith(rdflib.XSD.namespace) and o.datatype not in [rdflib.XSD.string, rdflib.XSD.boolean]:
            if p in numeric_data:
                numeric_data[p].append((s, o.value))
            else:
                numeric_data[p] = [(s, o.value)]
    return numeric_data

# 提取数值数据
numeric_data = extract_numeric_data()

# 打印结果
for p in numeric_data:
    print(p)
    for s, value in numeric_data[p]:
        print("\t", s, value)