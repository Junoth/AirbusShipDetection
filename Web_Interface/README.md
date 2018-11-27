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
Use **[MongoDB](https://www.mongodb.com/)** as the main database for the website.MongoDB is a document database with the scalability and flexibility that you want with the querying and indexing that you need.

You can go to the MongoDB website to download and install the MongoDB database.Use /data/db as the storage directory.

### Docker
To deploy on the server, we use **[Docker](https://www.docker.com/)** to make image of the website.docker is a popular application to help finish agile operations and integrated container security for legacy and cloud-native applications.It helps to deploy the website system quickly and safely.

After building the image,you just need to install docker on the server.Download the image and create the container.
