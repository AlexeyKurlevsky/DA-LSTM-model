FROM python:3.10.9

WORKDIR /code

RUN pip install --upgrade pip
RUN pip install poetry

COPY ./pyproject.toml ./poetry.lock /code/

RUN poetry config virtualenvs.create false
RUN poetry install $(test "$YOUR_ENV" == production && echo "--no-dev") --no-interaction --no-ansi

COPY ./src/ /code/src/
COPY ./app.py /code/

RUN mkdir -p ./upload_file

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0", "--port=8088"]