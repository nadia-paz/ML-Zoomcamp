import requests
import aws_url as au

#url = 'http://localhost:8080/2015-03-31/functions/function/invocations'

url = au.get_url()
data = {'url': 'http://bit.ly/mlbookcamp-pants'}

result = requests.post(url, json=data).json()
print(result)