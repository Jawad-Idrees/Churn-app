# Churn-app
This is a Python based application for churn prediction for telecom data. The front end is built with streamlit. Backend is FastAPI based. The data, model.pkl and the training.py files are also uploaded.
Here is the basic information of the data.
* The data is synthetic generated using numpy & pandas functions.
![image](https://github.com/user-attachments/assets/44ae44d8-dab3-4791-a9c9-1ff3b993c747)

The EDA reveals that most critical columns that relate to the Churn prediction are the numeric columns and a few nominal columns. Tenure of user's contract with the company & user's subscription type are importance.
* Dockerized app
  The application has been dockerize for one-click run and is available @ [jawadidrees/churn-app ](https://hub.docker.com/repository/docker/jawadidrees/churn-app)
* MLFlow has also been used for model experimentation. Here are a few screenshots.
  ![image](https://github.com/user-attachments/assets/102087f3-988c-419d-831d-b32838aceeb0)

* The best model with 97% accuracy.
  ![image](https://github.com/user-attachments/assets/8963c3cf-abda-41a3-97c3-e6dd48a66870)

