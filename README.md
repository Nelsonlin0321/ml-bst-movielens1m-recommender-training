# ml-bst-movielens1m-recommender-training


## Run It Locally

define env in .env file

```sh
AWS_DEFAULT_REGION=
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
ENV=dev
```

```sh
python -m venv venv
source venv/bin/activate
pip install requirements.txt
python train.py
```

## Run Using Docker Container
```sh
docker build . -t bst-movielens1m-recommender-training:latest  --platform linux/amd64
```

docker.env
```sh
AWS_DEFAULT_REGION=
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
ENV=dev
```

```sh
docker run -it \
        --env-file docker.env \
        --cpus=4 \
        --shm-size=4g bst-movielens1m-recommender-training:latest \
        --artifact_dir "./artifacts" \
        --model_save_dir "./models" \
        --sequence_length 6 \
        --test_size 0.85 \
        --genres_length 4 \
        --embedding_dim 32 \
        --dropout 0.2 \
        --epoches 5 \
        --learning_rate 0.001 \
        --batch_size 128
```

