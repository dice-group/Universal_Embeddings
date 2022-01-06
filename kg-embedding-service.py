import httpx
import json

# dev version!
# https://github.com/dice-group/kg-embedding-service
# https://www.python-httpx.org/
# https://docs.python.org/3/library/json.html

class KgEmbeddingService():

    def __init__(self, webservice_url='http://unikge.cs.upb.de:5001'):
        self.webservice_url = webservice_url

    def get_index_list(self):
        return json.loads(httpx.get(self.webservice_url + '/get-index-list').text)['index_list']

    def get_entity_embedding(self, indexname, entities):
        # POST is not allowed (405 Method Not Allowed)
        # GET can not contain JSON (see https://stackoverflow.com/a/983458)
        # return httpx.post(self.webservice_url + '/get-entity-embedding', json={'indexname':indexname,'entities':entities}).text
        # TODO: Example with different entities
        raise NotImplementedError

    def get_entity_embedding_neighbour(self):
        # /get-entity-embedding-neighbour
        raise NotImplementedError

    def get_entity_neighbour(self):
        # /get-entity-neighbour
        raise NotImplementedError

    def get_index_info(self, indexname):
        return httpx.post(self.webservice_url + '/get-index-info', json={'indexname':indexname}).text


#print(KgEmbeddingService().get_index_list())

# Not working now
#print(KgEmbeddingService().get_entity_embedding('shallom_dbpedia_index', '/resource/Boeing_747_hull_losses'))

#print(KgEmbeddingService().get_index_info('shallom_dbpedia_index'))
