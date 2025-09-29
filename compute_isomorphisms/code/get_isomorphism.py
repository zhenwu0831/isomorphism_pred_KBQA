from create_isomorphisms import *
from pprint import pprint

def obtain_isomorphism_id(s_expr):

    query_graph, G                    = get_query_graph(s_expr)
    curr_graph, qnid, tree_iso        = create_iso_graph(query_graph)


    if tree_iso:
        iso_num                       =f'T-{get_isomorphism_id(curr_graph, qnid, update=False)}'

    else:
        iso_num                       =f'C-{get_non_tree_isomorphism_id(curr_graph, qnid, update=False)}'

    return iso_num


def replace_s_expr(s_expr):

    query_graph, G                   = get_query_graph(s_expr)
    curr_graph, qnid, tree_iso       = create_iso_graph(query_graph)

    pprint(query_graph)

    if tree_iso:
        iso_num                         =f'T-{get_isomorphism_id(curr_graph, qnid, update=False)}'
    else:
        iso_num                         =f'C-{get_non_tree_isomorphism_id(curr_graph, qnid, update=False)}'

    new_s_expr                          = s_expr

    time_macro_date                     = 'time_macro 2015^^http://www.w3.org/2001/XMLSchema#date'
    new_s_expr                          = new_s_expr.replace(time_macro_date, '<TIME>')

    for edge in query_graph['edges']:
        new_s_expr                      = new_s_expr.replace(edge['relation'], '<RELATION>')

    for node in query_graph['nodes']:
        new_s_expr                      = new_s_expr.replace(node['id'], '<ENTITY>')

    new_s_expr                          = new_s_expr.replace('<ENTITY>.<TIME>', '<RELATION> <ENTITY>')

    print(new_s_expr)

    new_query_graph, new_G              = get_query_graph(new_s_expr)
    new_curr_graph, new_qnid, new_tree_iso  = create_iso_graph(new_query_graph)

    if new_tree_iso:
        new_iso_num                   =f'T-{get_isomorphism_id(new_curr_graph, new_qnid, update=False)}'
    else:
        new_iso_num                   =f'C-{get_non_tree_isomorphism_id(new_curr_graph, new_qnid, update=False)}'


    print(iso_num, new_iso_num)


# s_expr = "(JOIN (R government.government_position_held.office_holder) (AND (JOIN government.government_position_held.time_macro 2015^^http://www.w3.org/2001/XMLSchema#date) (AND (JOIN government.government_position_held.office_position_or_title m.0j5_3sz) (JOIN (R government.governmental_jurisdiction.governing_officials) m.06mkj))))"
# print(obtain_isomorphism_id(s_expr))

'''
s_expr_2="(JOIN (R government.government_position_held.office_holder) (TC (AND (JOIN (R government.governmental_jurisdiction.governing_officials) m.06mkj) (JOIN government.government_position_held.office_position_or_title m.0j5_3sz)) government.government_position_held.from NOW))",


print(obtain_isomorphism_id(s_expr_2))


print(replace_s_expr(s_expr))
'''

# print(replace_s_expr(s_expr_2))


def update_grailqa_generations():
    import pandas as pd
    import dill

    wandb_df = pd.read_csv("../../isomorphism_pred_KBQA/wandb_runs.csv")

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

        pred_sexps, true_sexps, y_trues = pred_data['pred_sexps'], pred_data['true_sexps'], pred_data['y_true']

        y_preds        = []

        for pred_sexp in tqdm(pred_sexps):
            try:
                iso_id = obtain_isomorphism_id(pred_sexp)
            except:
                iso_id = 'T-0'

            y_preds.append(iso_id)

        pred_data['isomorphism_from_sexpr'] = y_preds

        with open(f'{pred_dir}/grailqa_dev_predictions.json', 'w') as f:
            json.dump(pred_data, f, indent=4)    
    

# update_generations()


def update_grailqapp_generations():
    import pandas as pd
    import dill

    wandb_df = pd.read_csv("../../isomorphism_pred_KBQA/wandb_runs.csv")

    data_path = '/data/shire/projects/GenQA/ISOKBQA/isomorphism_pred_KBQA/data/grailqa/KBQA_walklet_edge_label_768_dim_data_test.dill'

    with open(data_path, "rb") as f:
        data = dill.load(f)

    test    =  data["test"]
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

        pred_sexps, true_sexps, y_trues = pred_data['pred_sexps'], pred_data['true_sexps'], pred_data['y_true']

        y_preds        = []

        for pred_sexp in tqdm(pred_sexps):
            try:
                iso_id = obtain_isomorphism_id(pred_sexp)
            except:
                iso_id = 'T-0'

            y_preds.append(iso_id)

        pred_data['isomorphism_from_sexpr'] = y_preds

        with open(f'{pred_dir}/grailqapp_test_predictions.json', 'w') as f:
            json.dump(pred_data, f, indent=4)    
    

# update_grailqapp_generations()