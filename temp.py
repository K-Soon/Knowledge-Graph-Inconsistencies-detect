from SPARQLWrapper import SPARQLWrapper, JSON

# 定义DBpedia命名空间前缀
PREFIX_DBPEDIA_OWL = "PREFIX dbpedia-owl: <http://dbpedia.org/ontology/>"

# 定义SPARQL查询语句
sparql = SPARQLWrapper("http://dbpedia.org/sparql")
sparql.setQuery(PREFIX_DBPEDIA_OWL +
"""
SELECT DISTINCT ?property WHERE {
  ?s rdf:type foaf:Person .
  ?s ?property ?o .
}
""")

# 设置返回格式为JSON
sparql.setReturnFormat(JSON)

# 执行查询并输出结果
results = sparql.query().convert()
for result in results["results"]["bindings"]:
    print(result["property"]["value"])
