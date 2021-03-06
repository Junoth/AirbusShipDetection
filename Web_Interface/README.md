Website
-

### Virtual environment(Recommended)
Python virtual environments are used to isolate package installation from the system.Configure your virtual environment 

with the tutorial in the [virtualenv](https://virtualenv.pypa.io/en/latest) website. 

If your virtualenv file name is venv,use the command below to turn into virtual environment.
```
. venv/bin/activate
```

### Web Frame
The **[website](http://www.airbusshipdect.online/)** for users has been set up.You can upload on this website and it will 
return the probability of ships.

**[Flask](http://flask.pocoo.org/)** is the main web frame for this website.Flask is a microframework for Python with two libraries:Werkzeug and Jinja 2.

![flask](http://flask.pocoo.org/static/logo/flask.png)

To setup flask,you need the command:
```
pip install Flask
```

Then,it can be easily used in the app:
```
from flask import Flask
```

### Web Server
The web server is **[Nginx](https://www.nginx.com/)**,which is a light-weight HTTP and reverse proxy server and a generic TCP/UDP proxy server.

For Linux system,you can use the command:
```
sudo apt-get install nginx
```
For MAC system
```
brew install nginx
```

### WSGI Server
Use **[uWSGI](https://uwsgi-docs.readthedocs.io/en/latest/#)** as the WSGI server, mainly used as the bridge between flask and nginx. 

Use the command to install uWSGI
```
apt-get install build-essential python
```

### Database
Use **[MongoDB](https://www.mongodb.com/)** as the main database for the website.MongoDB is a document database with the scalability and flexibility that you want with the querying and indexing that you need.You can go to the MongoDB website to download and install the MongoDB database.Use /data/db as the storage directory.

### Docker
Use **[Docker](https://www.docker.com/)** to deploy the website on server.docker is a popular application to help finish agile operations and integrated container security for legacy and cloud-native applications.It helps to deploy quickly and safely.

Here we use the [tiangolo/uwsgi-nginx-flask](https://hub.docker.com/r/tiangolo/uwsgi-nginx-flask/) image in the docker hub to make our own image.The Dockerfile is like below:
```
FROM tiangolo/uwsgi-nginx-flask:python2.7
COPY requirements.txt requirements.txt 
RUN pip install --no-cache-dir -r requirements.txt
COPY ./app /app
```

MongoDB is also deployed by docker by using the official docker image [mongo](https://hub.docker.com/_/mongo/)

Because of the sandbox mechanism,we need to use bridge network to connect web and mongodb.You can use docker-compose file directly to make containers.You can also use volumes to bind container with the local file.
```
version: '3'

services:
  web:
    image: web_v2:latest
    ports:
      - 8080:80
    networks:
      - my-net
  mongodb:
    image: mongo
    networks:
      - my-net
    ports:
      - 27017:27017
networks:
   my-net:
    driver: bridge   
```

### Origin Community Distribution
OCD is platform as a service PaaS provided by Red Hat.Use **[OCD](https://www.okd.io/)** to manage containers and load balance.OCD(Origin Community Distribution) provides a complete open source container application platform,which is built on **[Kubernetes](https://kubernetes.io/)**

First,install the Openshift origin.Use web-rs.yaml to create a replica of web pods and db-pod.yml to create a MongoDB pod,which is like below.

![openshift](https://raw.githubusercontent.com/Junoth/AirbusShipDetection/master/Web_Interface/images/openshift.png)

Use db-service.yaml to create a MongoDB service to make sure the web pod and database pod can connect with each other.The web-service.yaml is used to create the service to expose the web pod to external IP.Here we use NodePort.If you deploy the website on the server and the the cloud provide Load-Balancer service.Please use that.Then create a router to the web-pod.

### Load test
Here we use siege tool to simply test the website.We will simulate a High-Concurrency situation with 100 users.
```
siege -d 10 -c 100 http://168.122.216.101
```
We get the result as below.There are total 29511 hits and availability is 100%.The transaction rate can nearly access 300,which means our website can basicly meet our requirments.

![test_output](https://raw.githubusercontent.com/Junoth/AirbusShipDetection/master/Web_Interface/images/Loadtest.png)
