# python3
import requests
import json
import pandas as pd
import os
import time

end_point = 'http://127.0.0.1:5000/'
end_point_recommend = end_point + 'nsfw'
headers = {'Content-type': 'application/json'}


def nsfw(query):
    data = json.dumps(query)
    # print('data:',data)
    res = requests.post(end_point_recommend, data=data, headers=headers)
    # print(res.ok)
    res_json = res.json()
    # print(json.dumps(res_json, indent=2))
    return res_json


if __name__ == '__main__':
    query = {'filename': 'images/demo-card-1.jpg'}
    res_json = nsfw(query)
    print(res_json)
