import json
import requests
from kafka import KafkaConsumer

consumer = KafkaConsumer('test', bootstrap_servers=['localhost:9092'],
                         value_deserializer=lambda m: json.loads(m.decode('ascii')))
for message in consumer:
    value = message.value
    print("Recieved parameters:")
    print(value)
    headers = {
        'Content-Type':'application/json; charset=UTF-8',
        }
    info = requests.post("http://127.0.0.1:5000/predict",data=json.dumps(message.value['body']),headers=headers)
    print("The prediction is:")
    print(info.text)

