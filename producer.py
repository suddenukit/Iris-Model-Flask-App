import json
from kafka import KafkaProducer
import csv
import time

producer = KafkaProducer(bootstrap_servers=['localhost:9092'], 
                         value_serializer=lambda m: json.dumps(m).encode('ascii'))
while True:
    #info=csv.reader("test_set.csv")
    with open('test_set.csv', 'r') as f:
        reader = csv.reader(f)
        # Skip the 1st row (heading)
        next(reader)
        for row in reader:
            body={}
            body["sepal_length"]=row[1]
            body["sepal_width"]=row[2]
            body["petal_length"]=row[3]
            body["petal_width"]=row[4]
            producer.send('test', {"body": body})
            print('Sent successfully.')
            time.sleep(5)
producer.close()

