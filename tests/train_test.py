import os
import sys

curr_dir = os.path.dirname(__file__)
root_dir = os.path.dirname(curr_dir)
sys.path.insert(0, root_dir)


def test_train_model():

    from train import bst_movielens1m_recommender_training_pipeline

    bst_movielens1m_recommender_training_pipeline(env='test')
