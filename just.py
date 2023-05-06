from SPARQLWrapper import SPARQLWrapper, JSON

sparql = SPARQLWrapper("http://dbpedia.org/sparql")

# 查询多个人的出生日期和死亡日期
sparql.setQuery("""
    SELECT ?person ?birthdate ?deathdate 
    WHERE {
        ?person a dbo:Person .
        ?person dbo:birthDate ?birthdate .
        ?person dbo:deathDate ?deathdate .
       
        
        
    }
    LIMIT 500
""")

# 查询多个机构的创建日期和解散日期
#sparql.setQuery("""
#    SELECT ?organization ?founded ?dissolved
#    WHERE {
#        ?organization a dbo:Organisation .
#        OPTIONAL { ?organization dbo:foundingDate ?founded }
#        OPTIONAL { ?organization dbo:dissolutionDate ?dissolved }
#    }
#    LIMIT 100
#""")


# 将查询发送到SPARQL端点并获取结果
sparql.setReturnFormat(JSON)
results = sparql.query().convert()


lines=[]
# 处理结果，提取实体和日期属性的信息
for result in results["results"]["bindings"]:
    print(result)
    entity = result["person"]["value"]
    birthdate = result["birthdate"]["value"]



    deathdate=result["deathdate"]["value"]
    #print(f"{entity} born {birthdate}")
    lines.append(str(f"{entity} {birthdate} {deathdate}\n")[28:])
with open('people.txt', 'w',encoding='utf-8') as f:
    f.writelines(lines)



