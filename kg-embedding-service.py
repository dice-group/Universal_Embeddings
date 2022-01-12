# Python implementation for KG entity embedding Service
# https://github.com/dice-group/kg-embedding-service

# https://www.python-httpx.org/
# https://docs.python.org/3/library/json.html

import httpx
import json

class KgEmbeddingService():

    def __init__(self, webservice_url='http://unikge.cs.upb.de:5001'):
        self.webservice_url = webservice_url
        self.headers_json = { 'Content-Type': 'application/json' }

    def ping(self, seconds=1):
        try:
            return httpx.get(self.webservice_url + '/ping', timeout=seconds).status_code == 200
        except httpx.ConnectTimeout:
            return False

    # Indexes

    def get_index_list(self):
        return json.loads(httpx.get(self.webservice_url + '/get-index-list').text)['index_list']

    def get_index_info(self, indexname):
        return json.loads(httpx.post(self.webservice_url + '/get-index-info', json={'indexname':indexname}).text)

    # Entities

    def get_all_entity(self, indexname, size=0):
        # Note: size=0 means no limit and takes time
        data = json.JSONEncoder().encode({'indexname':indexname, 'size':size})
        response = httpx.post(self.webservice_url + '/get-all-entity', data=data, headers=self.headers_json)
        return json.loads(response.text)

    def get_entity_embedding(self, indexname, entities):
        # Single entity also has to be list
        if isinstance(entities, str):
            entities = [entities]
        data = json.JSONEncoder().encode({'indexname':indexname, 'entities':entities})
        response = httpx.post(self.webservice_url + '/get-entity-embedding', data=data, headers=self.headers_json)
        return json.loads(response.text)

    def get_entity_neighbour(self, indexname, entity):
        data = json.JSONEncoder().encode({'indexname':indexname, 'entity':entity})
        response = httpx.post(self.webservice_url + '/get-entity-neighbour', data=data, headers=self.headers_json)
        return json.loads(response.text)

    def get_embedding_neighbour(self, indexname, embedding, distmetric="cosine"):
        # Note: distmetric maybe not implemented now
        data = json.JSONEncoder().encode({'indexname':indexname, 'embedding':embedding, 'distmetric':distmetric})
        return json.loads(httpx.post(self.webservice_url + '/get-embedding-neighbour', data=data, headers=self.headers_json).text)

    # Relations

    def get_all_relation(self):
        # No documentation available -> under development
        # /get-all-relation
        raise NotImplementedError

    def get_relation_embedding(self):
        # No documentation available -> under development
        # /get-relation-embedding
        raise NotImplementedError

    def get_relation_neighbour(self):
        # No documentation available -> under development
        # /get-relation-neighbour
        raise NotImplementedError


# Examples
#
#response = KgEmbeddingService().ping(1)
#
#response = KgEmbeddingService().get_index_list()
#response = KgEmbeddingService().get_index_info('shallom_dbpedia_index')
#
#response = KgEmbeddingService().get_all_entity('shallom_dbpedia_index', size=10)
#response = KgEmbeddingService().get_entity_embedding('shallom_dbpedia_index', '/resource/Boeing_747_hull_losses')
#response = KgEmbeddingService().get_entity_embedding('shallom_dbpedia_index', ['/resource/Boeing_747_hull_losses', '/resource/Paderborn'])
#emb_pb=[0.2806874, -0.21256703, -0.47864893, 0.21929161, -0.2297767, -0.31447953, -0.1175671, 0.15006831, -0.10480619, -0.077421956, 0.56386936, 0.06708485, 0.55256176, -0.1584095, -0.115171954, -0.054179046, 0.35811296, 0.10366277, -0.3444387, 0.13747558, -0.29540744, -0.31908906, -0.25035203, -0.0570282, 0.27496752]
#response = KgEmbeddingService().get_entity_neighbour('shallom_dbpedia_index', '/resource/Paderborn')
#response = KgEmbeddingService().get_embedding_neighbour('shallom_dbpedia_index', emb_pb)
#
#print(type(response), response)

