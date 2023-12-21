import requests

class API:
    def __init__(self, baseURL):
        self.baseURL = baseURL

    def post(self, route, payload):
        # A POST request to the API
        post_response = requests.post(self.baseURL+'/'+route, json=payload)
        
        # Print the response
        post_response_json = post_response.json()
        print(post_response_json)

    def get(self, route):
        # A GET request to the API
        response = requests.get(self.baseURL+'/'+route)
        
        # Print the response
        response_json = response.json()
        print(response_json)
        return response_json
