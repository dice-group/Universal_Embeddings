import httpx
import json

# dev version!

class KgEmbeddingService():

    def __init__(self, webservice_url='http://unikge.cs.upb.de:5001'):
        self.webservice_url = webservice_url

    def get_index_list(self):
        r = httpx.get(self.webservice_url + '/get-index-list')
        data = json.loads(r.text)
        return data['index_list']

print(KgEmbeddingService().get_index_list())
