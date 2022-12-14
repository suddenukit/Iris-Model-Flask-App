# Iris-Model-Flask-App

I. Task requirements
1. Write a web application in the Python flask. The logical functions of the web application are as follows: a. According to the calyx length, calyx width, petal length and petal width to judge the flower species, the requirements are: i. Model training and deployment: download Iris dataset by yourself, train the model by yourself, and deploy the model in Web API mode through the Flask framework. ii. Model inference: After invoking the web-API in post mode, the result is returned.
Note: THE SAMPLES ARE divided INTO BIT TRAINING/VALIDATION sets, and the sample sets are divided for inference in STEP ii.
2. Send inference request based on Kafka message queue, specific requirements: a. Install and run kafka locally. b. Use python to simulate a data stream that can continuously send data to kafka. Each data is recorded as a sample in the Iris classification problem above, namely: calyx length, calyx width, petal length, petal width c. Write a consumer of the above data stream in python, ask the topic to update, call the API of task 1, and print the result after the API returns the result.


II. Solution Ideas
First, the data is trained as follows:
1. From UCI Iris dataset (http://archive.ics.uci.edu/ml/datasets/Iris) to download the Iris CSV data file. 
2. 2. Use Python to read the file and preprocess the data. 
3. 3. Split the data set randomly, extract the test set separately and save it for subsequent input. In this project, the ratio of training set, validation set and test set is 3:1:1. 
4. This project adopts sklearn's logistic regression to train the data. GridSearchCV is used to cross-validate the different parameters to get a good model. The cross-validation parameters include: 'c' : the inverse of the regularization coefficient; 'max_iter' : maximum number of iterations for algorithm convergence; 'class_weight' : type weight parameter; 'solver' : Optimization of the loss function. 
5. Output the accuracy of the current model on the test set and save the model. 
Then, deploy the model to local ports in Flask mode:
1. Create Flask and Api instances in another python program. 
2. Load the trained model. 
3. Inherit the Flask restful Resource module and create the post () method. 
4. Obtain attributes from user input and use the predict function to obtain the prediction result and return the result in JSON form. 
5. Use the add_resource method to load the written class to the specified api. 
Now that the first step of the task is complete, move on to the second step. 
1. Install and configure Kafka on the local PC and start Kafka and its Zookeeper dependencies. 
2. Create production and consumer Python programs with their instances connected to Kafka's default port 9092. 
3. The production end reads the saved test set, packages the test set line by line in JSON format, and sends the test set. After the data is successfully sent, wait for a period of time and then send the next row of data. The consumer receives the data, invokes the API in Task 1, and prints the output of the API.
