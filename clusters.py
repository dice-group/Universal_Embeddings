# Open in resource browser: http://fr.dbpedia.org/resource/Dookie
# -> http://fr.dbpedia.org/page/Dookie
#
# English object for <owl:sameAs>
# http://dbpedia.org/resource/Dookie
#
# Objects for <rdf:type>
# dbpedia-owl:Album
# dbpedia-owl:MusicalWork
# dbpedia-owl:Work

# SPARQL FR: http://fr.dbpedia.org/sparql
#
# select distinct ?type where { <http://fr.dbpedia.org/resource/Dookie> a ?type} LIMIT 100
#
# PREFIX res: <http://fr.dbpedia.org/resource/> SELECT ?uri ?type { ?uri a ?type FILTER (?uri IN (res:Dookie, res:Penny_Lane)) }


import json
import rdflib
import pickle
from os.path import exists
import collections

# Configuration
json_file = 'Clusters_Fr_universal.json'
cache_file = 'Clusters_Fr_universal.types.pickle'

# Read JSON file
f = open(json_file)
json = json.load(f)
f.close()

# Print JSON
if False:
    print(json.keys())

# Print clusters
if True:
    print('\nURIs in clusters:')
    for cluster in json.keys():
        print(cluster, len(json[cluster]))
if False:
    for cluster in json.keys():
        print(cluster)
        for uri in json[cluster]:
            print(uri)


# Get types
def query_types(uri_list):
    # Variants (inside SERVICE clause):
    # ?uri a ?type FILTER (?uri IN (res:Dookie, res:Penny_Lane))
    # ?uri a ?type FILTER (?uri IN (<http://fr.dbpedia.org/resource/Lady's_Bridge>, res:Penny_Lane))
    # ?uri a ?type FILTER (?uri IN (RESOURCES))
    types_query = """
        PREFIX res: <http://fr.dbpedia.org/resource/>
        SELECT ?uri ?type
        WHERE {
            SERVICE <http://fr.dbpedia.org/sparql> {
                ?uri a ?type FILTER (?uri IN (RESOURCES))
            } 
        }
    """

    uris = ['<' + sub for sub in uri_list]
    uris = [sub + '>' for sub in uris]
    types_query = types_query.replace('RESOURCES', ", ".join(uris))

    uris_to_types = {}
    for row in rdflib.Graph().query(types_query):
        uri_resource = str(row[0])
        uri_type = str(row[1])
        if uri_resource not in uris_to_types:
            uris_to_types[uri_resource] = []
        uris_to_types[uri_resource].append(uri_type)
    return uris_to_types


# Get types (and cache them in file)
# e.g. {'cluster-0': {'http://fr.dbpedia.org/resource/Beat_It': ['http://www.w3.org/2002/07/owl#Thing',
#       'http://schema.org/CreativeWork', 'http://dbpedia.org/ontology/MusicalWork', ...
types = {}
if not exists(cache_file):
    for cluster in json.keys():
        types[cluster] = query_types(json[cluster])
    pickle.dump(types, open(cache_file, 'wb'))
else:
    types = pickle.load(open(cache_file, 'rb'))

# Count types of each cluster
# e.g. OrderedDict([('cluster-0', {'http://www.w3.org/2002/07/owl#Thing': 139, 'http://schema.org/CreativeWork': 57,
#      'http://dbpedia.org/ontology/Work': 57, 'http://www.wikidata.org/entity/Q386724': 57, ...
counters = collections.OrderedDict()
for cluster in types:
    counter = {}
    for uri in types[cluster]:
        for type in types[cluster][uri]:
            counter[type] = counter.get(type, 0) + 1
    # https://stackoverflow.com/questions/613183/how-do-i-sort-a-dictionary-by-value
    counters[cluster] = dict(sorted(counter.items(), key=lambda item: item[1], reverse=True))

# Print top types for each cluster
# e.g. cluster-0 with 133 types
#      139 http://www.w3.org/2002/07/owl#Thing
#       57 http://schema.org/CreativeWork
#       57 http://dbpedia.org/ontology/Work
if True:
    print('\nTop types in clusters:')
    max = 10
    for cluster in counters:
        print(cluster, 'with', len(counters[cluster]), 'types')
        i = max
        for type in counters[cluster]:
            i -= 1
            print(counters[cluster][type], type)
            if i == 0:
                break

# Collect types used in every cluster
use_types = []
for cluster in counters:
    for type in counters[cluster]:
        type_usage = 0
        for cluster_check in counters:
            if type in counters[cluster_check]:
                type_usage += 1
        if type_usage < len(counters):
            use_types.append(type)


# Print top types for each cluster
# e.g. cluster-0 with 133 types
#      139 http://www.w3.org/2002/07/owl#Thing
#       57 http://schema.org/CreativeWork
#       57 http://dbpedia.org/ontology/Work
if True:
    print('\nTop types which do not occur in every cluster:')
    max = 5
    for cluster in counters:
        print(cluster)
        i = max
        for type in counters[cluster]:
            if type not in use_types:
                continue
            i -= 1
            print(counters[cluster][type], type)
            if i == 0:
                break