import os
import glob
import argparse
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn

from libs.dali_helper import J2kIterator, j2k_decode_pipeline, JllIterator, jll_process_pipeline, CustomDALIGenericIterator
from libs.image_processing import get_yolo
from libs.efficientnet import EffNetModel, predict

os.environ["CUDA_VISIBLE_DEVICES"] = '2'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_DIR = "/home/data4/share/rsna-breast-cancer-detection/test_images"

def main(args):
    df = pd.read_csv("/home/data4/share/rsna-breast-cancer-detection/test.csv")
    df['dcm'] = IMG_DIR + '/' + df.patient_id.astype(str) + '/' + df.image_id.astype(str) + '.dcm'
    df_j2k, df_jll = df[df["site_id"]==2].reset_index(drop=True), df[df["site_id"]==1].reset_index(drop=True)

    # build DALI pipeline and loader
    roi_model = get_yolo()
    j2k_it = J2kIterator(df_j2k, batch_size=32, img_dir=IMG_DIR)
    j2k_pipe = j2k_decode_pipeline(j2k_it, width=768, height=768, batch_size=32, num_threads=2, device_id=0, 
                                   py_num_workers=4, exec_async=False, exec_pipelined=False)
    j2k_loader = CustomDALIGenericIterator(yolo_model=roi_model, length=len(df_j2k), save_img=False, pipelines=[j2k_pipe])

    if len(df_jll) != 0:
        jll_it = JllIterator(df_jll, batch_size=32, img_dir=IMG_DIR)
        jll_pipe = jll_process_pipeline(jll_it, width=768, height=768, batch_size=32, num_threads=2, device_id=0, 
                                        py_num_workers=4, exec_async=False, exec_pipelined=False)
        jll_loader = CustomDALIGenericIterator(yolo_model=roi_model, length=len(df_jll), save_img=False, pipelines=[jll_pipe])

    # find checkpoints with lowest validation loss and highest AUC
    best_weights_vloss, best_weights_auc = [], []
    for i in range(4):
        min_vloss, best_weight_vloss = 10000.0, "vloss"
        max_auc, best_weight_auc = 0.5, "auc"
        for weight in glob.glob(f"ckpt/f{i}/*.pth"):
            if float(weight.split("_")[-1][:-4]) < min_vloss:
                min_vloss = float(weight.split("_")[-1][:-4])
                best_weight_vloss = weight
            if float(weight.split("_")[-3]) > max_auc:
                max_auc = float(weight.split("_")[-3])
                best_weight_auc = weight
        best_weights_vloss.append(best_weight_vloss)
        best_weights_auc.append(best_weight_auc)

    # build EfficientNet
    EffNets = []
    for weight in tqdm(best_weights_auc):
        model = EffNetModel()
        model.to(DEVICE)
        checkpoint = torch.load(weight, map_location=torch.device('cuda:0'))
        model.load_state_dict(checkpoint['state_dict_ema'])
        model.eval()
        EffNets.append(model)

    # make predictions 
    j2k_pred = predict(EffNets, j2k_loader)
    df_j2k['cancer'] = j2k_pred
    if len(df_jll)!=0:
        jll_pred = predict(EffNets, jll_loader)
        df_jll['cancer'] = jll_pred

    if len(df_jll) != 0:
        df = pd.concat([df_j2k, df_jll]).reset_index(drop=True)
    else:
        df = df_j2k
    df['prediction_id'] = df['patient_id'].astype(str) + '_' + df['laterality']
    sub = df[['prediction_id', "cancer"]].groupby("prediction_id").mean().reset_index()
    sub["cancer"] = (sub["cancer"] > args.threshold).astype(int)
    sub.to_csv('./output/submission.csv', index=False)
    print("All done, csv saved!")
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.01)
    args = parser.parse_args()
    main(args)