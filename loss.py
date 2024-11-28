import torch
from utils import Structure_Measure, Fbeta_Measure, MAE_Measure, IOU

BCE = torch.nn.BCEWithLogitsLoss()

def loss_func(feature_1, feature_2, feature_3, feature_4, feature_1_sig, feature_2_sig, feature_3_sig, feature_4_sig, gts):
    train_precision_4, train_recall_4, train_Fbeta_4 = Fbeta_Measure(feature_4_sig, gts)
    train_MAE = MAE_Measure(feature_4_sig, gts)
    train_Smearsure = Structure_Measure(feature_4_sig, gts)

    loss1 = BCE(feature_1, gts) + IOU(feature_1_sig, gts) + (1 - train_Fbeta_4)
    loss2 = BCE(feature_2, gts) + IOU(feature_2_sig, gts) + (1 - train_Fbeta_4)
    loss3 = BCE(feature_3, gts) + IOU(feature_3_sig, gts) + (1 - train_Fbeta_4)
    loss4 = BCE(feature_4, gts) + IOU(feature_4_sig, gts) + (1 - train_Fbeta_4)

    loss = loss1 + loss2 + loss3 + loss4
    return loss, loss4, train_precision_4, train_recall_4, train_Fbeta_4, train_MAE, train_Smearsure