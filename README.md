# ml-bst-movielens1m-recommender-training


## Run It Locally

define env in .env file

```sh
# Mlflow tracking url
TRACKING_URL=
# Mlflow experiment name
EXPERIMENT_NAME=bst-movielens1m-recommender-training
AWS_DEFAULT_REGION=
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
ENV=dev
```

```sh
python -m venv venv
source venv/bin/activate
pip install requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
python train.py --artifact_dir "./artifacts" \
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

## Run Using Docker Container
```sh
docker build . -t bst-movielens1m-recommender-training:latest  --platform linux/arm64/v8
```

```sh
docker run -it \
        --env-file .env \
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

