import pandas as pd
import numpy as np
from collections import defaultdict as ddict
import os
from tqdm import tqdm
import json
from pprint import pprint
import dill
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.calibration import calibration_curve
import argparse
from utils.evaluation import hits_at_k_text, hits_at_k_graph, compute_hits_at_k, compute_exact_match, compute_f1
import seaborn as sns

import matplotlib.pyplot as plt
from glob import glob
import torch
import seaborn as sns


def create_grailqa_iso():

    wandb_df = pd.read_csv("wandb_runs.csv")

    data_path = '/data/shire/projects/GenQA/ISOKBQA/isomorphism_pred_KBQA/data/grailqa/KBQA_walklet_edge_label_768_dim_data.dill'

    with open(data_path, "rb") as f:
        data = dill.load(f)

    train   = data["train"][:int(len(data["train"])*0.7)]
    dev     = data["train"][int(len(data["train"])*0.7):]
    test    = data["dev"]

    levels  = [elem['level'] for elem in test]

    results_dict = ddict(list)

    wandb_df = wandb_df[(wandb_df['State'] == 'finished') & (wandb_df['tasks']=='iso')]

    good_cnt = 0

    for idx, row in wandb_df.iterrows():
        
        pred_dir       = row['output_dir'].replace('ckpts_rdutt', "predictions")        

        if not os.path.exists(f'{pred_dir}/grailqa_dev_predictions.json'):
            print(f"Skipping {pred_dir} as it does not exist")
            continue

        pred_data      = json.load(open(f'{pred_dir}/grailqa_dev_predictions.json', 'r'))    

        y_true, y_pred = pred_data['y_true'], pred_data['y_pred']

        results_dict['seed'].append(row['seed'])
        results_dict['model_type'].append(row['model_type'])
        results_dict['gnn_model'].append(row['gnn_model'])
        results_dict['hidden_dim'].append(row['hidden_dim'])
        results_dict['gnn_depth'].append(row['gnn_depth'])
        results_dict['isomorphism'].append('ALL')
        results_dict['level'].append('ALL')
        results_dict['F1'].append(f1_score(y_true, y_pred, average='macro'))

        # results_dict['iso_f1'].append(f1_score(y_true, y_pred, average='macro'))

        class_f1_scores = f1_score(y_true, y_pred, average=None)

        for i, score in enumerate(class_f1_scores):
            results_dict['seed'].append(row['seed'])
            results_dict['model_type'].append(row['model_type'])
            results_dict['gnn_model'].append(row['gnn_model'])
            results_dict['hidden_dim'].append(row['hidden_dim'])
            results_dict['gnn_depth'].append(row['gnn_depth'])
            results_dict['isomorphism'].append(f'T-{i}')
            results_dict['level'].append('NA')
            results_dict['F1'].append(score)

            # results_dict[f'ISO-{i}'].append(score)
        
        for level in set(levels):
            level_y_true = [y for y, l in zip(y_true, levels) if l == level]
            level_y_pred = [y for y, l in zip(y_pred, levels) if l == level]
            level_f1 = f1_score(level_y_true, level_y_pred, average='macro')
            # results_dict[f'level_{level}_f1'].append(level_f1)
            results_dict['seed'].append(row['seed'])
            results_dict['model_type'].append(row['model_type'])
            results_dict['gnn_model'].append(row['gnn_model'])
            results_dict['hidden_dim'].append(row['hidden_dim'])
            results_dict['gnn_depth'].append(row['gnn_depth'])
            results_dict['isomorphism'].append('NA')
            results_dict['level'].append(level)
            results_dict['F1'].append(level_f1)
        
        print(f'Completed run {row["wandb_run"]}')
        good_cnt += 1

    print(f"Processed {good_cnt} good runs out of {len(wandb_df)}")

    results_df = pd.DataFrame(results_dict)
    results_df.to_csv("results/iso_results_grailqa_summary.csv", index=False)



def create_grailqapp_iso():

    wandb_df = pd.read_csv("wandb_runs.csv")

    data_path = '/data/shire/projects/GenQA/ISOKBQA/isomorphism_pred_KBQA/data/grailqa/KBQA_walklet_edge_label_768_dim_data_test.dill'

    with open(data_path, "rb") as f:
        data = dill.load(f)

    test    = data["test"]

    levels  = [elem['level'] for elem in test]

    results_dict = ddict(list)

    wandb_df = wandb_df[(wandb_df['State'] == 'finished') & (wandb_df['tasks']=='iso')]

    good_cnt = 0

    for idx, row in wandb_df.iterrows():
        
        pred_dir       = row['output_dir'].replace('ckpts_rdutt', "predictions")        

        if not os.path.exists(f'{pred_dir}/grailqapp_test_predictions.json'):
            print(f"Skipping {pred_dir} as it does not exist")
            continue

        pred_data      = json.load(open(f'{pred_dir}/grailqapp_test_predictions.json', 'r'))    

        y_true, y_pred = pred_data['y_true'], pred_data['y_pred']

        results_dict['seed'].append(row['seed'])
        results_dict['model_type'].append(row['model_type'])
        results_dict['gnn_model'].append(row['gnn_model'])
        results_dict['hidden_dim'].append(row['hidden_dim'])
        results_dict['gnn_depth'].append(row['gnn_depth'])
        results_dict['isomorphism'].append('ALL')
        results_dict['level'].append('ALL')
        results_dict['F1'].append(f1_score(y_true, y_pred, average='macro'))

        # results_dict['iso_f1'].append(f1_score(y_true, y_pred, average='macro'))

        class_f1_scores = f1_score(y_true, y_pred, average=None)

        for i, score in enumerate(class_f1_scores):
            results_dict['seed'].append(row['seed'])
            results_dict['model_type'].append(row['model_type'])
            results_dict['gnn_model'].append(row['gnn_model'])
            results_dict['hidden_dim'].append(row['hidden_dim'])
            results_dict['gnn_depth'].append(row['gnn_depth'])
            results_dict['isomorphism'].append(f'T-{i}')
            results_dict['level'].append('NA')
            results_dict['F1'].append(score)

            # results_dict[f'ISO-{i}'].append(score)
        
        for level in set(levels):
            level_y_true = [y for y, l in zip(y_true, levels) if l == level]
            level_y_pred = [y for y, l in zip(y_pred, levels) if l == level]
            level_f1 = f1_score(level_y_true, level_y_pred, average='macro')
            # results_dict[f'level_{level}_f1'].append(level_f1)
            results_dict['seed'].append(row['seed'])
            results_dict['model_type'].append(row['model_type'])
            results_dict['gnn_model'].append(row['gnn_model'])
            results_dict['hidden_dim'].append(row['hidden_dim'])
            results_dict['gnn_depth'].append(row['gnn_depth'])
            results_dict['isomorphism'].append('NA')
            results_dict['level'].append(level)
            results_dict['F1'].append(level_f1)
        
        print(f'Completed run {row["wandb_run"]}')
        good_cnt += 1

    print(f"Processed {good_cnt} good runs out of {len(wandb_df)}")

    results_df = pd.DataFrame(results_dict)
    results_df.to_csv("results/grailqapp_iso_results_summary.csv", index=False)




def plot_grailqa_iso_charts():

    results_df = pd.read_csv("results/iso_results_summary.csv")

    results_df = results_df[(results_df['hidden_dim']==128) & (results_df['gnn_depth']==2) & (results_df['seed'].isin([15232, 98109, 123456, 987654, 15213]))]

    print(len(results_df))

    # PLOT THE BEST SCORES FOR GRAPH AND COD MODELS

    graph_df    = results_df[(results_df['model_type']=='graph')]

    curr_df     = graph_df[graph_df['isomorphism'] != 'NA']
    sns.set(style="whitegrid", font_scale=1.5)
    sns.catplot(data=curr_df, x= 'isomorphism', y ='F1', hue='gnn_model', height=5, aspect=1.5, kind='bar')
    plt.suptitle('Isomorphism-wise F1 scores of Graph models', fontsize=16, y=1.05)
    plt.savefig('results/iso_results_graph_isomorphism_wise.png', bbox_inches='tight')
    plt.clf()
    plt.close()

    curr_df     = graph_df[graph_df['level'] != 'NA']
    sns.set(style="whitegrid", font_scale=1.5)
    sns.catplot(data=curr_df, x= 'level', y ='F1', hue='gnn_model', height=5, aspect=1.5, kind='bar')
    plt.suptitle('Level-wise F1 scores of Graph models', fontsize=16, y=1.05)
    plt.savefig('results/iso_results_graph_level_wise.png', bbox_inches='tight')
    plt.clf()
    plt.close()


    cod_df      = results_df[(results_df['model_type']=='cod')]

    curr_df     = cod_df[cod_df['isomorphism'] != 'NA']
    sns.set(style="whitegrid", font_scale=1.5)
    sns.catplot(data=curr_df, x= 'isomorphism', y ='F1', hue='gnn_model', height=5, aspect=1.5, kind='bar')
    plt.suptitle('Isomorphism-wise F1 scores of COD models', fontsize=16, y=1.05)
    plt.savefig('results/iso_results_cod_isomorphism_wise.png', bbox_inches='tight')
    plt.clf()
    plt.close()

    curr_df     = cod_df[cod_df['level'] != 'NA']
    sns.set(style="whitegrid", font_scale=1.5)
    sns.catplot(data=curr_df, x= 'level', y ='F1', hue='gnn_model', height=5, aspect=1.5, kind='bar')
    plt.suptitle('Level-wise F1 scores of COD models', fontsize=16, y=1.05)
    plt.savefig('results/iso_results_cod_level_wise.png', bbox_inches='tight')
    plt.clf()
    plt.close()


    text_df    = results_df[(results_df['model_type']=='text')]

    curr_df     = text_df[text_df['isomorphism'] != 'NA']
    sns.set(style="whitegrid", font_scale=1.5)
    sns.catplot(data=curr_df, x= 'isomorphism', y ='F1', height=5, aspect=1.5, kind='bar')
    plt.suptitle('Isomorphism-wise F1 scores of text models', fontsize=16, y=1.05)
    plt.savefig('results/iso_results_text_isomorphism_wise.png', bbox_inches='tight')
    plt.clf()
    plt.close()

    curr_df     = text_df[text_df['level'] != 'NA']
    sns.set(style="whitegrid", font_scale=1.5)
    sns.catplot(data=curr_df, x= 'level', y ='F1', height=5, aspect=1.5, kind='bar')
    plt.suptitle('Level-wise F1 scores of text models', fontsize=16, y=1.05)
    plt.savefig('results/iso_results_text_level_wise.png', bbox_inches='tight')
    plt.clf()
    plt.close()


    # compare best text and cod models

    best_cod_df = cod_df[cod_df['gnn_model']=='rgat']
    best_gnn_df = graph_df[graph_df['gnn_model']=='gat']
    best_df     = pd.concat([best_cod_df, best_gnn_df, text_df])

    
    curr_df     = best_df[best_df['isomorphism'] != 'NA']
    sns.set(style="whitegrid", font_scale=1.5)
    sns.catplot(data=curr_df, x= 'isomorphism', y ='F1', hue='model_type', height=5, aspect=1.5, kind='bar')
    plt.suptitle('Isomorphism-wise F1 scores of Best Models', fontsize=16, y=1.05)
    plt.savefig('results/iso_results_models_isomorphism_wise.png', bbox_inches='tight')
    plt.clf()
    plt.close()

    curr_df     = best_df[best_df['level'] != 'NA']
    sns.set(style="whitegrid", font_scale=1.5)
    sns.catplot(data=curr_df, x= 'level', y ='F1', hue='model_type', height=5, aspect=1.5, kind='bar')
    plt.suptitle('Level-wise F1 scores of Best Models', fontsize=16, y=1.05)
    plt.savefig('results/iso_results_models_level_wise.png', bbox_inches='tight')
    plt.clf()
    plt.close()



def plot_grailqapp_iso_charts():

    results_df = pd.read_csv("results/iso_results_grailqapp_summary.csv")

    results_df = results_df[(results_df['hidden_dim']==128) & (results_df['gnn_depth']==2) & (results_df['seed'].isin([15232, 98109, 123456, 987654, 15213]))]

    print(len(results_df))

    # PLOT THE BEST SCORES FOR GRAPH AND COD MODELS

    graph_df    = results_df[(results_df['model_type']=='graph')]

    curr_df     = graph_df[graph_df['isomorphism'] != 'NA']
    sns.set(style="whitegrid", font_scale=1.5)
    sns.catplot(data=curr_df, x= 'isomorphism', y ='F1', hue='gnn_model', height=5, aspect=1.5, kind='bar')
    plt.suptitle('Isomorphism-wise F1 scores of Graph models', fontsize=16, y=1.05)
    plt.savefig(f'results/{args.dataset}_iso_results_graph_isomorphism_wise.png', bbox_inches='tight')
    plt.clf()
    plt.close()


    cod_df      = results_df[(results_df['model_type']=='cod')]

    curr_df     = cod_df[cod_df['isomorphism'] != 'NA']
    sns.set(style="whitegrid", font_scale=1.5)
    sns.catplot(data=curr_df, x= 'isomorphism', y ='F1', hue='gnn_model', height=5, aspect=1.5, kind='bar')
    plt.suptitle('Isomorphism-wise F1 scores of COD models', fontsize=16, y=1.05)
    plt.savefig(f'results/{args.dataset}_iso_results_cod_isomorphism_wise.png', bbox_inches='tight')
    plt.clf()
    plt.close()

    text_df    = results_df[(results_df['model_type']=='text')]

    curr_df     = text_df[text_df['isomorphism'] != 'NA']
    sns.set(style="whitegrid", font_scale=1.5)
    sns.catplot(data=curr_df, x= 'isomorphism', y ='F1', height=5, aspect=1.5, kind='bar')
    plt.suptitle('Isomorphism-wise F1 scores of text models', fontsize=16, y=1.05)
    plt.savefig(f'results/{args.dataset}_iso_results_text_isomorphism_wise.png', bbox_inches='tight')
    plt.clf()
    plt.close()


    # compare best text and cod models

    best_cod_df = cod_df[cod_df['gnn_model']=='rgat']
    best_gnn_df = graph_df[graph_df['gnn_model']=='gat']
    best_df     = pd.concat([best_cod_df, best_gnn_df, text_df])
    
    import pdb; pdb.set_trace()

    curr_df     = best_df[best_df['isomorphism'] != 'NA']
    sns.set(style="whitegrid", font_scale=1.5)
    sns.catplot(data=curr_df, x= 'isomorphism', y ='F1', hue='model_type', height=5, aspect=1.5, kind='bar')
    plt.suptitle('Isomorphism-wise F1 scores of Best Models', fontsize=16, y=1.05)
    plt.savefig(f'results/{args.dataset}_iso_results_models_isomorphism_wise.png', bbox_inches='tight')
    plt.clf()
    plt.close()




def create_grailqa_generation():

    wandb_df = pd.read_csv("wandb_runs.csv")

    data_path = '/data/shire/projects/GenQA/ISOKBQA/isomorphism_pred_KBQA/data/grailqa/KBQA_walklet_edge_label_768_dim_data.dill'

    with open(data_path, "rb") as f:
        data = dill.load(f)

    train   = data["train"][:int(len(data["train"])*0.7)]
    dev     = data["train"][int(len(data["train"])*0.7):]
    test    = data["dev"]

    levels  = [elem['level'] for elem in test]

    results_dict = ddict(list)

    wandb_df = wandb_df[(wandb_df['State'] == 'finished') & (wandb_df['tasks']=='generation')]

    good_cnt = 0

    for idx, row in wandb_df.iterrows():
        
        pred_dir       = row['output_dir'].replace('ckpts_rdutt', "predictions")        

        if not os.path.exists(f'{pred_dir}/grailqa_dev_predictions.json'):
            print(f"Skipping {pred_dir} as it does not exist")
            continue

        pred_data      = json.load(open(f'{pred_dir}/grailqa_dev_predictions.json', 'r'))    

        pred_sexps, true_sexps, y_trues = pred_data['pred_sexps'], pred_data['true_sexps'], pred_data['y_true']

        for pred_sexp, gold_sexp, iso_true, level in zip(pred_sexps, true_sexps, y_trues, levels):
            em = compute_exact_match(pred_sexp, gold_sexp)
            results_dict['seed'].append(row['seed'])
            results_dict['model_type'].append(row['model_type'])
            results_dict['gnn_model'].append(row['gnn_model'])
            results_dict['hidden_dim'].append(row['hidden_dim'])
            results_dict['gnn_depth'].append(row['gnn_depth'])
            results_dict['isomorphism'].append(iso_true)
            results_dict['level'].append(level)
            results_dict['EM'].append(em)

            if 'MTL' in row['Name']:
                results_dict['MTL'].append('Yes')
            else:
                results_dict['MTL'].append('No')

            results_dict['seed'].append(row['seed'])
            results_dict['model_type'].append(row['model_type'])
            results_dict['gnn_model'].append(row['gnn_model'])
            results_dict['hidden_dim'].append(row['hidden_dim'])
            results_dict['gnn_depth'].append(row['gnn_depth'])
            results_dict['isomorphism'].append('ALL')
            results_dict['level'].append('ALL')
            results_dict['EM'].append(em)

            if 'MTL' in row['Name']:
                results_dict['MTL'].append('Yes')
            else:
                results_dict['MTL'].append('No')

        print(f'Completed run {row["wandb_run"]}')
        good_cnt += 1

    print(f"Processed {good_cnt} good runs out of {len(wandb_df)}")

    results_df = pd.DataFrame(results_dict)
    results_df.to_csv("results/grailqa_generation_results_summary.csv", index=False)




def create_grailqapp_generation():

    wandb_df = pd.read_csv("wandb_runs.csv")

    data_path = '/data/shire/projects/GenQA/ISOKBQA/isomorphism_pred_KBQA/data/grailqa/KBQA_walklet_edge_label_768_dim_data_test.dill'

    with open(data_path, "rb") as f:
        data = dill.load(f)

    test = data["test"]

    levels  = [elem['level'] for elem in test]

    results_dict = ddict(list)

    wandb_df = wandb_df[(wandb_df['State'] == 'finished') & (wandb_df['tasks']=='generation')]

    good_cnt = 0

    for idx, row in wandb_df.iterrows():
        
        pred_dir       = row['output_dir'].replace('ckpts_rdutt', "predictions")        

        if not os.path.exists(f'{pred_dir}/grailqapp_test_predictions.json'):
            print(f"Skipping {pred_dir} as it does not exist")
            continue

        pred_data      = json.load(open(f'{pred_dir}/grailqapp_test_predictions.json', 'r'))    

        pred_sexps, true_sexps, y_trues = pred_data['pred_sexps'], pred_data['true_sexps'], pred_data['y_true']

        for pred_sexp, gold_sexp, iso_true, level in zip(pred_sexps, true_sexps, y_trues, levels):
            em = compute_exact_match(pred_sexp, gold_sexp)
            results_dict['seed'].append(row['seed'])
            results_dict['model_type'].append(row['model_type'])
            results_dict['gnn_model'].append(row['gnn_model'])
            results_dict['hidden_dim'].append(row['hidden_dim'])
            results_dict['gnn_depth'].append(row['gnn_depth'])
            results_dict['isomorphism'].append(iso_true)
            results_dict['level'].append(level)
            results_dict['EM'].append(em)

            if 'MTL' in row['Name']:
                results_dict['MTL'].append('Yes')
            else:
                results_dict['MTL'].append('No')

            results_dict['seed'].append(row['seed'])
            results_dict['model_type'].append(row['model_type'])
            results_dict['gnn_model'].append(row['gnn_model'])
            results_dict['hidden_dim'].append(row['hidden_dim'])
            results_dict['gnn_depth'].append(row['gnn_depth'])
            results_dict['isomorphism'].append('ALL')
            results_dict['level'].append('ALL')
            results_dict['EM'].append(em)

            if 'MTL' in row['Name']:
                results_dict['MTL'].append('Yes')
            else:
                results_dict['MTL'].append('No')

        print(f'Completed run {row["wandb_run"]}')
        good_cnt += 1

    print(f"Processed {good_cnt} good runs out of {len(wandb_df)}")

    results_df = pd.DataFrame(results_dict)
    results_df.to_csv("results/grailqapp_generation_results_summary.csv", index=False)






def plot_grailqa_gen_charts():

    results_df = pd.read_csv("results/generation_results_summary.csv")

    results_df = results_df[results_df['seed'].isin([15232, 98109, 123456, 987654, 15213])]

    print(len(results_df))

    # PLOT THE EM scores for the generation models

    sns.set(style="whitegrid", font_scale=1.5)
    sns.catplot(data=results_df, x= 'level', y ='EM', hue='MTL', height=5, aspect=1.5, kind='bar', order=sorted(results_df['level'].unique()))
    plt.suptitle('Level-wise EM scores of text/MTL models', fontsize=16, y=1.05)
    plt.savefig('results/grailqa_gen_results_level_wise.png', bbox_inches='tight')
    plt.clf()
    plt.close()


    sns.set(style="whitegrid", font_scale=1.5)
    sns.catplot(data=results_df, x= 'isomorphism', y ='EM', hue='MTL', height=5, aspect=1.5, kind='bar', order=sorted(results_df['isomorphism'].unique()))
    plt.suptitle('Isomorphism-wise EM scores of text/MTL models', fontsize=16, y=1.05)
    plt.savefig('results/grailqa_gen_results_isomorphism_wise.png', bbox_inches='tight')
    plt.clf()
    plt.close()




def plot_grailqapp_gen_charts():

    results_df = pd.read_csv("results/grailqapp_generation_results_summary.csv")

    results_df = results_df[results_df['seed'].isin([15232, 98109, 123456, 987654, 15213])]

    print(len(results_df))

    # PLOT THE EM scores for the generation models

    sns.set(style="whitegrid", font_scale=1.5)
    sns.catplot(data=results_df, x= 'isomorphism', y ='EM', hue='MTL', height=5, aspect=1.5, kind='bar', order=sorted(results_df['isomorphism'].unique()))
    plt.suptitle('Isomorphism-wise EM scores of text/MTL models', fontsize=16, y=1.05)
    plt.savefig('results/grailqapp_gen_results_isomorphism_wise.png', bbox_inches='tight')
    plt.clf()
    plt.close()




def grailqa_iso_from_generation():

    wandb_df = pd.read_csv("wandb_runs.csv")

    data_path = '/data/shire/projects/GenQA/ISOKBQA/isomorphism_pred_KBQA/data/grailqa/KBQA_walklet_edge_label_768_dim_data.dill'

    with open(data_path, "rb") as f:
        data = dill.load(f)

    train   = data["train"][:int(len(data["train"])*0.7)]
    dev     = data["train"][int(len(data["train"])*0.7):]
    test    = data["dev"]

    levels  = [elem['level'] for elem in test]

    results_dict = ddict(list)

    wandb_df = wandb_df[(wandb_df['State'] == 'finished') & (wandb_df['tasks']=='generation')]

    good_cnt = 0

    for idx, row in wandb_df.iterrows():
        if 'MTL' in row['Name']:
            continue
        
        pred_dir       = row['output_dir'].replace('ckpts_rdutt', "predictions")        

        if not os.path.exists(f'{pred_dir}/grailqa_dev_predictions.json'):
            print(f"Skipping {pred_dir} as it does not exist")
            continue

        pred_data      = json.load(open(f'{pred_dir}/grailqa_dev_predictions.json', 'r'))    

        

        pred_sexps, true_sexps, y_trues, y_preds = pred_data['pred_sexps'], pred_data['true_sexps'], pred_data['y_true'], pred_data['isomorphism_from_sexpr']

        for pred_sexp, gold_sexp, iso_true, iso_pred, level in zip(pred_sexps, true_sexps, y_trues, y_preds, levels):
            results_dict['seed'].append(row['seed'])
            results_dict['iso_true'].append(iso_true)
            results_dict['iso_pred'].append(iso_pred)
            results_dict['level'].append(level)        


        print(f'Completed run {row["wandb_run"]}')
        good_cnt += 1

    print(f"Processed {good_cnt} good runs out of {len(wandb_df)}")

    results_df = pd.DataFrame(results_dict)

    

    seeds   = results_df['seed'].unique()
    levels  = results_df['level'].unique()
    isos    = results_df['iso_true'].unique()

    iso_dict = ddict(list)
    for level in levels:
        
        y_pred = results_df[(results_df['level']==level)]['iso_pred'].tolist()
        y_true = results_df[(results_df['level']==level)]['iso_true'].tolist()

        f1 = f1_score(y_true, y_pred, average='macro')
        if np.isnan(f1):
            f1 = 0.0

        iso_dict['level'].append(level)
        iso_dict['F1'].append(f1)

        print(f'F1 score for level {level} computed is {f1}.')
    
    #### passing through isomorphisms #####

    y_pred = results_df['iso_pred'].tolist()
    y_true = results_df['iso_true'].tolist()

    f1 = f1_score(y_true, y_pred, average='macro')
    if np.isnan(f1):
        f1 = 0.0

    iso_dict['level'].append('ALL')
    iso_dict['F1'].append(f1)

    print(f'F1 score for isomorphism computed is {f1}.')

    iso_df = pd.DataFrame(iso_dict)

    sns.set(style="whitegrid", font_scale=1.5)
    sns.catplot(data=iso_df, x= 'level', y ='F1' , height=5, aspect=1.5, kind='bar', order=sorted(iso_df['level'].unique())) 
    plt.suptitle('Level-wise F1 scores of generated s-expressions', fontsize=16, y=1.05)
    plt.savefig('results/gen_results_isof1_level_wise.png', bbox_inches='tight')
    plt.clf()
    plt.close()


    y_pred = results_df['iso_pred'].tolist()
    y_true = results_df['iso_true'].tolist()

    f1_arr = f1_score(y_true, y_pred, average=None, labels= isos)

    iso_dict = ddict(list)

    for iso, f1 in zip(isos, f1_arr):
        iso_dict['isomorphism'].append(iso)
        iso_dict['F1'].append(f1)

        print(f'F1 score for isomorphism {iso} computed is {f1}.')
    
    iso_dict['isomorphism'].append('ALL')
    f1 = f1_score(y_true, y_pred, average='macro')
    iso_dict['F1'].append(f1)

    iso_df = pd.DataFrame(iso_dict)

    sns.set(style="whitegrid", font_scale=1.5)
    sns.catplot(data=iso_df, x= 'isomorphism', y ='F1' , height=5, aspect=1.5, kind='bar', order=sorted(iso_df['isomorphism'].unique())) 
    plt.suptitle('Isomorphism-wise F1 scores of generated s-expressions', fontsize=16, y=1.05)
    plt.savefig('results/gen_results_isof1_isomorphism_wise.png', bbox_inches='tight')
    plt.clf()
    plt.close()



    
    # results_df.to_csv("results/generation_results_summary.csv", index=False)



def grailqapp_iso_from_generation():

    wandb_df = pd.read_csv("wandb_runs.csv")

    data_path = '/data/shire/projects/GenQA/ISOKBQA/isomorphism_pred_KBQA/data/grailqa/KBQA_walklet_edge_label_768_dim_data_test.dill'

    with open(data_path, "rb") as f:
        data = dill.load(f)

    test   = data["test"]

    levels  = [elem['level'] for elem in test]

    results_dict = ddict(list)

    wandb_df = wandb_df[(wandb_df['State'] == 'finished') & (wandb_df['tasks']=='generation')]

    good_cnt = 0

    for idx, row in wandb_df.iterrows():
        if 'MTL' in row['Name']:
            continue
        
        pred_dir       = row['output_dir'].replace('ckpts_rdutt', "predictions")        

        if not os.path.exists(f'{pred_dir}/grailqapp_test_predictions.json'):
            print(f"Skipping {pred_dir} as it does not exist")
            continue

        pred_data      = json.load(open(f'{pred_dir}/grailqapp_test_predictions.json', 'r'))    
        

        pred_sexps, true_sexps, y_trues, y_preds = pred_data['pred_sexps'], pred_data['true_sexps'], pred_data['y_true'], pred_data['isomorphism_from_sexpr']

        for pred_sexp, gold_sexp, iso_true, iso_pred, level in zip(pred_sexps, true_sexps, y_trues, y_preds, levels):
            results_dict['seed'].append(row['seed'])
            results_dict['iso_true'].append(iso_true)
            results_dict['iso_pred'].append(iso_pred)
            results_dict['level'].append(level)        


        print(f'Completed run {row["wandb_run"]}')
        good_cnt += 1

    print(f"Processed {good_cnt} good runs out of {len(wandb_df)}")

    results_df = pd.DataFrame(results_dict)

    # import pdb; pdb.set_trace()
    

    seeds   = results_df['seed'].unique()
    isos    = results_df['iso_true'].unique()
    
    #### passing through isomorphisms #####

    # import pdb; pdb.set_trace()

    y_pred = results_df['iso_pred'].tolist()
    y_true = results_df['iso_true'].tolist()

    f1_arr = f1_score(y_true, y_pred, labels = isos, average=None)

    iso_dict = ddict(list)

    for iso, f1 in zip(isos, f1_arr):
        iso_dict['isomorphism'].append(iso)
        iso_dict['F1'].append(f1)

        print(f'F1 score for isomorphism {iso} computed is {f1}.')
    
    iso_dict['isomorphism'].append('ALL')
    f1 = f1_score(y_true, y_pred, average='macro')
    iso_dict['F1'].append(f1)

    iso_df = pd.DataFrame(iso_dict)

    sns.set(style="whitegrid", font_scale=1.5)
    sns.catplot(data=iso_df, x= 'isomorphism', y ='F1' , height=5, aspect=1.5, kind='bar', order=sorted(iso_df['isomorphism'].unique())) 
    plt.suptitle('Isomorphism-wise F1 scores of generated s-expressions', fontsize=16, y=1.05)
    plt.savefig('results/grailqapp_gen_results_isof1_isomorphism_wise.png', bbox_inches='tight')
    plt.clf()
    plt.close()



    
    # results_df.to_csv("results/generation_results_summary.csv", index=False)



def make_webqsp_results():

    
    webqsp_data = json.load(open('data/webqsp/test_shuffled.json', 'r'))

    id2level = {}

    for elem in webqsp_data:
        id2level[elem['ID']] = elem['level']

    results_dict = ddict(list)

    result_files = glob('results/webqsp_RPP_results/*.json')

    seedwise_results = ddict(dict)

    for file in result_files:

        seed = file.split('/')[-1].split('_')[-1][:-5]

        pred_data = json.load(open(file, 'r'))

        if 'no-mulco' in file:
            model_type = 'Hybrid + No COD'
            continue
        elif 'unified' in file:
            model_type = 'CoD'
        elif 't5-base' in file:
            model_type = 'Text'
        elif 'gnn' in file:
            model_type = 'Graph'

        y_true, y_pred = [], []

        for elem in pred_data:

            qid     = elem['id']
            level   = id2level[qid]

            results_dict['model_type'].append(model_type)
            results_dict['level'].append(level)
            results_dict['seed'].append(seed)
            
            if 'true' in elem:
                results_dict['isomorphism'].append(f'T-{elem["true"]}')
                y_true.append(elem['true'])
            if 'label' in elem:
                results_dict['isomorphism'].append(f'T-{elem["label"]}')
                y_true.append(elem['label'])
            
            if 'predicted' in elem:
                results_dict['predicted'].append(f'T-{elem["predicted"]}')
                y_pred.append(elem['predicted'])
            if 'prediction' in elem:
                results_dict['predicted'].append(f'T-{elem["prediction"]}')
                y_pred.append(elem['prediction'])

        seedwise_results[model_type][seed] = f1_score(y_true, y_pred, average='macro')
    

    best_k = 3
    best_seedwise_dict = ddict(list)

    for model_type in seedwise_results:

        seed_f1_dict = seedwise_results[model_type]

        sorted_seeds = sorted(seed_f1_dict.items(), key=lambda x: x[1], reverse=True)

        if best_k !='All':
            topk_seeds = [seed for seed, f1 in sorted_seeds[:best_k]]
        else:
            topk_seeds = [seed for seed, f1 in sorted_seeds]

        print(f'Top sorted seeds for model {model_type} are {topk_seeds}')

        for seed in topk_seeds:
            best_seedwise_dict[model_type].append(seed)
            
    ## choose the top k seeds 

    results_df      = pd.DataFrame(results_dict)

    answers_dict    = ddict(list)

    for model_type in results_df['model_type'].unique():

        for level in results_df['level'].unique():
            
            level_df = results_df[(results_df['model_type']==model_type) & (results_df['level']==level) & results_df['seed'].isin(best_seedwise_dict[model_type])]
            y_true = level_df['isomorphism'].tolist()
            y_pred = level_df['predicted'].tolist()
            f1 = f1_score(y_true, y_pred, average='macro')
            answers_dict['model_type'].append(model_type)
            answers_dict['level'].append(level)
            answers_dict['F1'].append(f1)
            print(f'F1 score for level {level} computed is {f1}.')

        model_df = results_df[(results_df['model_type']==model_type) & results_df['seed'].isin(best_seedwise_dict[model_type])]
        y_true = model_df['isomorphism'].tolist()
        y_pred = model_df['predicted'].tolist()
        f1 = f1_score(y_true, y_pred, average='macro')
        answers_dict['model_type'].append(model_type)
        answers_dict['level'].append('ALL')
        answers_dict['F1'].append(f1)
        print(f'F1 score for model {model_type} ALL computed is {f1}.')
    
    levels_df = pd.DataFrame(answers_dict)
    sns.set(style="whitegrid", font_scale=1.5)
    sns.catplot(data=levels_df, x= 'level', y ='F1', hue='model_type', height=5, aspect=1.5, kind='bar')
    plt.suptitle('Level-wise F1 scores on WebQSP', fontsize=16, y=1.05)
    plt.savefig(f'results/webqsp_level_wise_{best_k}_nohybrid.png', bbox_inches='tight')
    plt.clf()
    plt.close()
    

    answers_dict    = ddict(list)

    for model_type in results_df['model_type'].unique():

        model_df = results_df[(results_df['model_type']== model_type)& results_df['seed'].isin(best_seedwise_dict[model_type])]

        y_true = model_df['isomorphism'].tolist()
        y_pred = model_df['predicted'].tolist()
        f1_arr = f1_score(y_true, y_pred, average=None)

        for i, f1 in enumerate(f1_arr):
            iso = f'T-{i}'
            answers_dict['model_type'].append(model_type)
            answers_dict['isomorphism'].append(iso)
            answers_dict['F1'].append(f1)

            print(f'F1 score for model {model_type} isomorphism {iso} computed is {f1}.')
            
        f1 = f1_score(y_true, y_pred, average='macro')
        answers_dict['model_type'].append(model_type)
        answers_dict['isomorphism'].append('ALL')
        answers_dict['F1'].append(f1)
        print(f'F1 score for model {model_type} ALL computed is {f1}.')
    
    iso_df = pd.DataFrame(answers_dict)
    sns.set(style="whitegrid", font_scale=1.5)
    sns.catplot(data=iso_df, x= 'isomorphism', y ='F1', hue='model_type', height=5, aspect=1.5, kind='bar')
    plt.suptitle('Isomorphism-wise F1 scores on WebQSP', fontsize=16, y=1.05)
    plt.savefig(f'results/webqsp_isomorphism_wise_{best_k}_nohybrid.png', bbox_inches='tight')
    plt.clf()
    plt.close()



def agg_grailqa_isos():

    wandb_df = pd.read_csv("wandb_runs.csv")

    # data_path = '/data/shire/projects/GenQA/ISOKBQA/isomorphism_pred_KBQA/data/grailqa/KBQA_walklet_edge_label_768_dim_data.dill'
    data_path = '/data/shire/projects/GenQA/ISOKBQA/isomorphism_pred_KBQA/data/grailqa/KBQA_walklet_edge_label_768_dim_data.dill'

    with open(data_path, "rb") as f:
        data = dill.load(f)

    test    = data["dev"]

    levels  = [elem['level'] for elem in test]

    wandb_df = wandb_df[(wandb_df['State'] == 'finished') & (wandb_df['tasks']=='iso')& (wandb_df['model_type']=='cod') & (wandb_df['gnn_model']=='rgat')]

    good_cnt = 0

    results_dict = ddict(lambda: ddict(lambda: ddict(dict)))

    import pdb; pdb.set_trace()

    print('Total runs found:', len(wandb_df))

    for idx, row in wandb_df.iterrows():
        
        pred_dir       = row['output_dir'].replace('ckpts_rdutt', "predictions")        

        if not os.path.exists(f'{pred_dir}/grailqapp_test_predictions.json'):
            print(f"Skipping {pred_dir} as it does not exist")
            continue

        pred_data      = json.load(open(f'{pred_dir}/grailqa_dev_predictions.json', 'r'))    

        y_true, y_pred = pred_data['y_true'], pred_data['y_pred']
        

        model_type = row['model_type']

        if model_type != 'text':
            gnn_val    = f'{row["gnn_model"]}_{row["hidden_dim"]}_{row["gnn_depth"]}'
        else:
            gnn_val    = 'NA'

        seed       = row['seed']
        results_dict[model_type][gnn_val][seed]['y_true'] = y_true
        results_dict[model_type][gnn_val][seed]['y_pred'] = y_pred

        print(f'Completed run {row["wandb_run"]}')
        good_cnt += 1


    agg_dict = ddict(list)

    for model_type in results_dict:
        for gnn_val in results_dict[model_type]:

            agg_y_true = ddict(lambda: ddict(int))
            agg_y_pred = ddict(lambda: ddict(int))

            try:
                assert len(results_dict[model_type][gnn_val]) == 5
            except AssertionError:
                continue

            for seed in results_dict[model_type][gnn_val]:

                curr_y_pred = results_dict[model_type][gnn_val][seed]['y_pred']

                for idx, pred in enumerate(curr_y_pred):
                    agg_y_pred[idx][f'T-{pred}'] += 1

                curr_y_true = results_dict[model_type][gnn_val][seed]['y_true']
                for idx, true in enumerate(curr_y_true):
                    agg_y_true[idx][f'T-{true}'] += 1

            best_y_true = [max(agg_y_true[idx].items(), key=lambda x: x[1])[0] for idx in range(len(agg_y_true))]
            best_y_pred = [max(agg_y_pred[idx].items(), key=lambda x: x[1])[0] for idx in range(len(agg_y_pred))]

            print('F1 score for model_type:', model_type, 'gnn_val:', gnn_val, 'is', f1_score(best_y_true, best_y_pred, average='macro'))

            agg_file = open(f'agg_predictions/{model_type}_{gnn_val}_grailq_dev_predictions.json','w')

            dump_data = []

            # import pdb; pdb.set_trace()

            idx       = 0
            for elem, pred_iso in zip(test, best_y_pred):
                # import pdb; pdb.set_trace()
                curr_data = {}
                curr_data['ID'] = elem['ID']
                curr_data['pred_isomorphism'] = agg_y_pred[idx]
                curr_data['true_isomorphism'] = elem['isomorphism']
                curr_data['question']         = elem['question']
                curr_data['level']            = elem['level']

                idx +=1
                
                dump_data.append(curr_data)
            
            json.dump(dump_data, agg_file, indent=4)



def agg_grailqapp_isos():

    wandb_df = pd.read_csv("wandb_runs.csv")

    data_path = '/data/shire/projects/GenQA/ISOKBQA/isomorphism_pred_KBQA/data/grailqa/KBQA_walklet_edge_label_768_dim_data_test.dill'

    with open(data_path, "rb") as f:
        data = dill.load(f)

    test    = data["test"]

    levels  = [elem['level'] for elem in test]

    wandb_df = wandb_df[(wandb_df['State'] == 'finished') & (wandb_df['tasks']=='iso')]

    good_cnt = 0

    results_dict = ddict(lambda: ddict(lambda: ddict(dict)))

    print('Total runs found:', len(wandb_df))

    for idx, row in wandb_df.iterrows():
        
        pred_dir       = row['output_dir'].replace('ckpts_rdutt', "predictions")        

        if not os.path.exists(f'{pred_dir}/grailqapp_test_predictions.json'):
            print(f"Skipping {pred_dir} as it does not exist")
            continue

        pred_data      = json.load(open(f'{pred_dir}/grailqapp_test_predictions.json', 'r'))    

        y_true, y_pred = pred_data['y_true'], pred_data['y_pred']

        model_type = row['model_type']

        if model_type != 'text':
            gnn_val    = f'{row["gnn_model"]}_{row["hidden_dim"]}_{row["gnn_depth"]}'
        else:
            gnn_val    = 'NA'

        seed       = row['seed']
        results_dict[model_type][gnn_val][seed]['y_true'] = y_true
        results_dict[model_type][gnn_val][seed]['y_pred'] = y_pred

        print(f'Completed run {row["wandb_run"]}')
        good_cnt += 1


    agg_dict = ddict(list)

    for model_type in results_dict:
        for gnn_val in results_dict[model_type]:

            agg_y_true = ddict(lambda: ddict(int))
            agg_y_pred = ddict(lambda: ddict(int))

            try:
                assert len(results_dict[model_type][gnn_val]) == 5
            except AssertionError:
                continue

            for seed in results_dict[model_type][gnn_val]:

                curr_y_pred = results_dict[model_type][gnn_val][seed]['y_pred']

                for idx, pred in enumerate(curr_y_pred):
                    agg_y_pred[idx][pred] += 1

                curr_y_true = results_dict[model_type][gnn_val][seed]['y_true']
                for idx, true in enumerate(curr_y_true):
                    agg_y_true[idx][true] += 1

            best_y_true = [max(agg_y_true[idx].items(), key=lambda x: x[1])[0] for idx in range(len(agg_y_true))]
            best_y_pred = [max(agg_y_pred[idx].items(), key=lambda x: x[1])[0] for idx in range(len(agg_y_pred))]

            print('F1 score for model_type:', model_type, 'gnn_val:', gnn_val, 'is', f1_score(best_y_true, best_y_pred, average='macro'))





def agg_grailqapp_isos_gen():

    wandb_df = pd.read_csv("wandb_runs.csv")

    data_path = '/data/shire/projects/GenQA/ISOKBQA/isomorphism_pred_KBQA/data/grailqa/KBQA_walklet_edge_label_768_dim_data_test.dill'

    with open(data_path, "rb") as f:
        data = dill.load(f)

    test   = data["test"]

    levels  = [elem['level'] for elem in test]

    results_dict = ddict(list)

    wandb_df = wandb_df[(wandb_df['State'] == 'finished') & (wandb_df['tasks']=='generation')]

    good_cnt = 0

    seedwise_results = ddict(lambda: ddict(list))

    for idx, row in wandb_df.iterrows():
        if 'MTL' not in row['Name']:
            continue
        
        pred_dir       = row['output_dir'].replace('ckpts_rdutt', "predictions")        

        if not os.path.exists(f'{pred_dir}/grailqapp_test_predictions.json'):
            print(f"Skipping {pred_dir} as it does not exist")
            continue

        pred_data      = json.load(open(f'{pred_dir}/grailqapp_test_predictions.json', 'r'))    

        y_trues, y_preds = pred_data['y_true'], pred_data['y_pred']

        for iso_true, iso_pred in zip(y_trues, y_preds):

            seedwise_results[row['seed']]['y_true'].append(iso_true)
            seedwise_results[row['seed']]['y_pred'].append(iso_pred)

    agg_y_true = ddict(lambda: ddict(int))
    agg_y_pred = ddict(lambda: ddict(int))

    for seed in seedwise_results.keys():

        curr_y_pred = seedwise_results[seed]['y_pred']

        for idx, pred in enumerate(curr_y_pred):
            agg_y_pred[idx][pred] += 1

        curr_y_true = seedwise_results[seed]['y_true']
        for idx, true in enumerate(curr_y_true):
            agg_y_true[idx][true] += 1

    best_y_true = [max(agg_y_true[idx].items(), key=lambda x: x[1])[0] for idx in range(len(agg_y_true))]
    best_y_pred = [max(agg_y_pred[idx].items(), key=lambda x: x[1])[0] for idx in range(len(agg_y_pred))]

    print('F1 scores', f1_score(best_y_true, best_y_pred, average='macro'))
    print('Accuracy', accuracy_score(best_y_true, best_y_pred))

    agg_file = open('agg_predictions/grailqapp_MTL_predictions.json','w')

    dump_data = []

    idx       = 0
    for elem, pred_iso in zip(data['test'], best_y_pred):
        curr_data = {}
        curr_data['ID'] = elem['ID']
        curr_data['pred_isomorphism'] = agg_y_pred[idx]
        curr_data['true_isomorphism'] = elem['isomorphism']
        curr_data['question']         = elem['question']

        idx +=1
        
        dump_data.append(curr_data)
    
    json.dump(dump_data, agg_file, indent=4)



def get_ece():

    import torch

    wandb_df = pd.read_csv("wandb_runs.csv")

    if args.dataset == 'grailqa':
        data_path = '/data/shire/projects/GenQA/ISOKBQA/isomorphism_pred_KBQA/data/grailqa/KBQA_walklet_edge_label_768_dim_data.dill'
    if args.dataset == 'grailqapp':
        data_path = '/data/shire/projects/GenQA/ISOKBQA/isomorphism_pred_KBQA/data/grailqa/KBQA_walklet_edge_label_768_dim_data_test.dill'

    with open(data_path, "rb") as f:
        data = dill.load(f)

    if args.dataset == 'grailqa':
        test    = data["dev"]
    if args.dataset == 'grailqapp':
        test    = data["test"]

    iso_count = ddict(int)

    for elem in test:
        iso_count[elem['isomorphism']] += 1
    
    pprint(iso_count)

    levels  = [elem['level'] for elem in test]

    results_dict = ddict(list)
    good_cnt = 0
    

    wandb_df = wandb_df[(wandb_df['State'] == 'finished') & (wandb_df['tasks']=='iso') & (wandb_df['model_type']=='cod')& (wandb_df['gnn_model']=='rgat')]

    all_logits = []
    all_labels = []
    all_preds  = []        

    for idx, row in wandb_df.iterrows():
        
        pred_dir       = row['output_dir'].replace('ckpts_rdutt', "predictions")        

        if args.dataset == 'grailqapp':
            pred_data      = json.load(open(f'{pred_dir}/grailqapp_test_predictions_with_isologits.json', 'r'))    
        if args.dataset == 'grailqa':
            pred_data      = json.load(open(f'{pred_dir}/grailqa_dev_predictions.json', 'r'))

        # iso_logits.argmax(-1).cpu().tolist()

        iso_labels, iso_preds, iso_logits = pred_data['y_true'], pred_data['y_pred'], pred_data['iso_logits']
        all_logits.extend(iso_logits)
        all_labels.extend(iso_labels)
        all_preds.extend(iso_preds)

    ### Generate also confusion matrices for the different isomorphism classes ###

    print('Overall accuracy is', accuracy_score(all_labels, all_preds))
    print('Overall F1 score is', f1_score(all_labels, all_preds, average='macro'))
    print('Overall F1 score is', f1_score(all_labels, all_preds, average=None))

    cm = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=[f'T-{i}' for i in range(cm.shape[0])], yticklabels=[f'T-{i}' for i in range(cm.shape[0])])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Normalized Confusion Matrix for {args.dataset}')
    plt.savefig(f'results/{args.dataset}_confusion_matrix.png', bbox_inches='tight')
    plt.clf()
    plt.close()
    # purge the file




    ### aggregated the data and logits across all instances ####
    all_logits = torch.tensor(all_logits)  # (N, C)
    probs = torch.softmax(all_logits, dim=1)  # (N, C)
    N, C = probs.shape

    ### Classwise ECE ###

    n_bins = 10
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)

    ece_classwise = torch.zeros(C)

    labels = torch.tensor(all_labels)

    calibration_dict = ddict(list)

    for c in range(C):
        probs_c = probs[:, c]
        labels_c = (labels == c).float()

        ece_c = torch.zeros(1)

        for i in range(n_bins):
            lower, upper = bin_boundaries[i], bin_boundaries[i+1]
            mask = (probs_c > lower) & (probs_c <= upper)

            if mask.sum() > 0:
                acc_bin = labels_c[mask].mean()       # true freq of class c
                conf_bin = probs_c[mask].mean()       # mean predicted prob for class c
                weight_bin = mask.float().mean()      # fraction of samples
                ece_c += weight_bin * torch.abs(acc_bin - conf_bin)

                calibration_dict['class'].append(f'T-{c}')
                calibration_dict['acc'].append(acc_bin.item())
                calibration_dict['conf'].append(conf_bin.item())

        ece_classwise[c] = ece_c.item()
        print(f'Class {c} : Freq {labels_c.sum().item()//5} ECE: {ece_classwise[c]:.4f}')
    
    calibration_df = pd.DataFrame(calibration_dict)
    calibration_df.to_csv(f'results/{args.dataset}_classwise_calibration.csv', index=False)

    sns.set(style="whitegrid", font_scale=1.2)
    sns.lineplot(data=calibration_df, x='conf', y='acc', hue='class', marker='o')
    # change the aspect ratio of the figure to be more wide than tall

    # make the figure wider than tall
    plt.gcf().set_size_inches(12, 8)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title(f'Class-wise Calibration Plot for {args.dataset}')
    plt.savefig(f'results/{args.dataset}_classwise_calibration_plot.png', bbox_inches='tight')
    plt.clf()
    plt.close()



        
    
    


if __name__ =='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--step",       required=True, help="Step to run")
    parser.add_argument("--task",       default='iso', help="iso")
    parser.add_argument("--dataset",    required=True, default="grailqa", help="dataset to use")


    args = parser.parse_args()
    
    if args.step == 'create':
        if args.task == 'iso':
            if args.dataset == 'grailqa':
                create_grailqa_iso()
            elif args.dataset == 'grailqapp':
                create_grailqapp_iso()
            
        elif args.task == 'gen':
            if args.dataset == 'grailqa':
                create_grailqa_generation()
            elif args.dataset == 'grailqapp':
                create_grailqapp_generation()
    

    elif args.step == 'agg':
        if args.task == 'iso':
            if args.dataset == 'grailqapp':
                agg_grailqapp_isos()
            elif args.dataset == 'grailqa':
                agg_grailqa_isos()

        elif args.task == 'iso_gen':
            if args.dataset == 'grailqapp':
                agg_grailqapp_isos_gen()
        


    elif args.step == 'plot':
        if args.task == 'iso':
            if args.dataset == 'grailqa':
                plot_grailqa_iso_charts()
            elif args.dataset == 'grailqapp':
                plot_grailqapp_iso_charts()

                    
        elif args.task == 'gen':
            if args.dataset == 'grailqa':
                plot_grailqa_gen_charts()
            elif args.dataset == 'grailqapp':
                plot_grailqapp_gen_charts()
        

        elif args.task == 'iso_gen':
            if args.dataset == 'grailqa':
                grailqa_iso_from_generation()
            elif args.dataset == 'grailqapp':
                grailqapp_iso_from_generation()
    

    elif args.step == 'ece':
        if args.task == 'iso':
            get_ece()
            
        # carry out error calibration analysis

