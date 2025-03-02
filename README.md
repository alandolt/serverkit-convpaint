![EPFL Center for Imaging logo](https://imaging.epfl.ch/resources/logo-for-gitlab.svg)
# serverkit-serverkit-convpaint

Implementation of a web server for [Convpaint](https://github.com/guiwitz/napari-convpaint).

Author: Alex Landolt

## Installing the algorithm server with `pip`

Install dependencies:

```
pip install -r requirements.txt
```

Run the server:

```
uvicorn main:app --host 0.0.0.0 --port 8000
```

or run `python main.py`. The server will be running on http://localhost:8000.

## Endpoints

A documentation of the endpoints is automatically generated at http://localhost:8000/docs.

**GET endpoints**

- http://localhost:8000/ : Running algorithm message.
- http://localhost:8000/version : Version of the `imaging-server-kit` package.
- http://localhost:8000/serverkit-convpaint/info : Web page displaying project metadata.
- http://localhost:8000/serverkit-convpaint/demo : Plotly Dash web demo app.
- http://localhost:8000/serverkit-convpaint/parameters : Json Schema representation of algorithm parameters.
- http://localhost:8000/serverkit-convpaint/sample_images : Byte string representation of the sample images.

**POST endpoints**

- http://localhost:8000/serverkit-convpaint/process : Endpoint to run the algorithm.

## Running the server with `docker-compose`

To build the docker image and run a container for the algorithm server in a single command, use:

```
docker compose up
```

The server will be running on http://localhost:8000.

## Running the server with `docker`

Build the docker image:

```
docker build --build-arg PYTHON_VERSION=3.11 -t serverkit-serverkit-convpaint .
```

Run the server in a container:

```
docker run -it --rm -p 8000:8000 serverkit-serverkit-convpaint:latest
```

The server will be running on http://localhost:8000.

## Running unit tests

If you have implemented unit tests in the [tests/](./tests/) folder, you can run them using pytest:

```
pytest
```

if you are developing your server locally, or

```
docker run --rm serverkit-serverkit-convpaint:latest pytest
```

to run the tests in a docker container.

<!-- ## Sample images provenance -->

<!-- Fill if necessary. -->
