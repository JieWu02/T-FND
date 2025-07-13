import random

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from data.data_utils import make_dfs, build_loaders
from .evaluation import *
# Import batch_constructor locally to avoid circular import
from models.fake_news_model import FakeNewsModel, calculate_loss


def test(config, test_loader, trial_number=None):

    try:
        checkpoint = torch.load(str(config.output_path) + '/checkpoint_' + str(trial_number) + '.pt')
    except:
        checkpoint = torch.load(str(config.output_path) + '/checkpoint.pt')

    try:
        parameters = checkpoint['parameters']
        config.assign_hyperparameters(parameters)
    except:
        pass

    model = FakeNewsModel(config).to(config.device)
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except:
        model.load_state_dict(checkpoint)

    model.eval()

    torch.manual_seed(27)
    random.seed(27)
    np.random.seed(27)

    image_features = []
    text_features = []
    multimodal_features = []
    concat_features = []

    targets = []
    predictions = []
    scores = []
    ids = []
    losses = []
    similarities = []
    tqdm_object = tqdm(test_loader, total=len(test_loader))
    for i, batch in enumerate(tqdm_object):
        from training.learner import batch_constructor
        batch = batch_constructor(config, batch)
        with torch.no_grad():
            evidence, evidence_a, _, _ = model(batch, 50) 
            loss, c_loss, s_loss = calculate_loss(model)

            _, output = torch.max(evidence_a.data, 1)

            predictions.append(output.detach())
            scores.append(evidence_a.detach())

            targets.append(batch['label'].detach())
            ids.append(batch['id'].detach())
            image_features.append(model.image_embeddings.detach())
            text_features.append(model.text_embeddings.detach())
            multimodal_features.append(model.multimodal_embeddings.detach())
            # concat_features.append(model.classifier.embeddings.detach())
            # similarities.append(model.similarity.detach())
            losses.append((loss.detach(), loss.detach(), s_loss.detach()))

    s = ''
    s += report_per_class(targets, predictions) + '\n'
    s += metrics(targets, predictions, scores, file_path=str(config.output_path) + '/fpr_tpr.csv') + '\n'
    with open(config.output_path + '/results.txt', 'w') as f:
        f.write(s)

    roc_auc_plot(targets, scores, fname=str(config.output_path) + "/roc.png")
    precision_recall_plot(targets, scores, fname=str(config.output_path) + "/pr.png")

    save_embedding(image_features, fname=str(config.output_path) + '/image_features.tsv')
    save_embedding(text_features, fname=str(config.output_path) + '/text_features.tsv')
    save_embedding(multimodal_features, fname=str(config.output_path) + '/multimodal_features_.tsv')
    save_embedding(concat_features, fname=str(config.output_path) + '/concat_features.tsv')
    save_2D_embedding(similarities, fname=str(config.output_path))

    config_parameters = str(config)
    with open(config.output_path + '/parameters.txt', 'w') as f:
        f.write(config_parameters)

    save_loss(ids, predictions, targets, losses, str(config.output_path) + '/text_label.csv')



def test_main(config, trial_number=None):
    train_df, test_df, validation_df = make_dfs(config, )
    test_loader = build_loaders(config, test_df, mode="test")
    test(config, test_loader, trial_number)
