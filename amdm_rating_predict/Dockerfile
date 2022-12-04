#FROM python:3.8
FROM ubuntu
LABEL maintainer="vokoramus01@gmail.com"
RUN apt-get update -y
RUN apt-get install -y python-pip
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8180
EXPOSE 8181
VOLUME /app/models
COPY ./docker-entrypoint.sh /
RUN chmod +x /docker-entrypoint.sh
ENTRYPOINT ["/docker-entrypoint.sh"]
