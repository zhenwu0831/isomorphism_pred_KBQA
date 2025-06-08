from collections import defaultdict as ddict
import networkx as nx
from networkx.algorithms.isomorphism import rooted_tree_isomorphism
import json, os, pickle, dill
from pprint import pprint
from tqdm import tqdm
import matplotlib.pyplot as plt
from create_sem_matcher import *
from logic_form_util import *
from sparql_executor import *
from pyvis.network import Network
import matplotlib.pyplot as plt
import pandas as pd
import networkx.algorithms.isomorphism as nx_iso
from pyvis.network import Network
import matplotlib.pyplot as plt

isomorphisms            = []
grail_iso_cnt           = ddict(list)
correct_grail_iso_cnt   = ddict(list)

tree_isomorphisms       = []
non_tree_isomorphisms   = []

def get_isomorphism_id(G, root, update= False):
    for idx, iso in enumerate(tree_isomorphisms):
        iso_G, iso_root = iso['graph'], iso['root']    
        poss_isos = rooted_tree_isomorphism(iso_G, iso_root, G, root)
        if len(poss_isos) >0:
            return idx      
    if update:  
        tree_isomorphisms.append({'graph': G, 'root':root})
        return len(tree_isomorphisms)-1
    else:
        None

## get the corresponding non-tree isomorphism id given the graph
def get_non_tree_isomorphism_id(G, root, update= False):
    for idx, iso in enumerate(non_tree_isomorphisms):
        iso_G = iso['graph']
        is_iso = nx.isomorphism.is_isomorphic(iso_G, G, node_match= nx_iso.categorical_node_match(['color','question_node'],['red',0]) )
        if is_iso:
            return idx
        
    if update:  
        non_tree_isomorphisms.append({'graph': G, 'root':root })
        return len(non_tree_isomorphisms)-1
    else:
        None


## actually create the graph from the given graph object

def create_iso_graph(graph):
    nodes, edges    = graph['nodes'], graph['edges']
    node_list       = []
    qnid            = 0
    tree_iso        = True

    for node in nodes:
        ntype, nid, qnode = node['node_type'], node['nid'], node['question_node']

        if qnode == 1: 
            ncolor = 'green'
            qnid   = nid
        elif qnode ==0 and ntype != 'class':
            ncolor = 'red'
        else:
            ncolor = 'grey'
        
        node_list.append((nid, {'color': ncolor, 'node_type': ntype, 'question_node': qnode }))
    
    edge_list       = []
    for edge in edges:
        edge_list.append((edge['start'], edge['end']))
    
    curr_graph      = nx.Graph()
    curr_graph.add_nodes_from(node_list)
    curr_graph.add_edges_from(edge_list)

    curr_nodes = curr_graph.nodes(data=True)
    curr_edges = curr_graph.edges(data=True)

    try:
        assert len(curr_nodes) == len(curr_edges)+1
    except Exception as e:
        tree_iso = False


    return curr_graph, qnid, tree_iso



def lisp_to_nested_expression(lisp_string: str) -> List:
    """
    Takes a logical form as a lisp string and returns a nested list representation of the lisp.
    For example, "(count (division first))" would get mapped to ['count', ['division', 'first']].
    """
    stack: List = []
    current_expression: List = []
    tokens = lisp_string.split()
    for token in tokens:
        while token[0] == '(':
            nested_expression: List = []
            current_expression.append(nested_expression)
            stack.append(current_expression)
            current_expression = nested_expression
            token = token[1:]
        current_expression.append(token.replace(')', ''))
        while token[-1] == ')':
            current_expression = stack.pop()
            token = token[:-1]
    return current_expression[0]


def get_query_graph(s_expr):

    
    query_graph                 =     {"nodes":[], "edges":[]}
    expr_list                   =     lisp_to_nested_expression(s_expr)
    G                           =     logical_form_to_graph(expr_list)    

    
    for node in G.nodes(data=True):
        nid , ndata = node[0], node[1]
        node_type   = ndata['type']
        id_         = ndata['id']
        # fname       = get_friendly_name(id_)
        if "question_node" in ndata:
            qnode   = 1
        else:
            qnode   = 0
        
        query_graph['nodes'].append({'nid': nid-1, 'node_type': node_type, 'function':'none',\
            'id': id_, 'class': "FILLER", 'question_node': qnode })
        
    for edge in G.edges(data=True):
        
        start, end, rel = edge[0]-1, edge[1]-1, edge[2]['relation']
        # fname           = get_friendly_name(rel)
        
        query_graph['edges'].append({'start': start, 'end': end, 'relation': rel})
    
    return query_graph, G


def process_datasets_for_isomorphism():

    datasets = ['grailqa', 'webqsp', 'graphquestions', 'cwq']

    splits  = {
                'webqsp': ['train', 'test'],
                'grailqa': ['train', 'test', 'dev'],
                'cwq': ['train', 'test', 'dev'],
                'graphquestions': ['train', 'test']
            }

    # predict the isomorphism for each of the query graph
    for dataset in datasets:
        for split in splits[dataset]:

            data                                      = json.load(open(f'../input_data/{dataset}.{split}.json'))

            for line in tqdm(data):
                try:
                    graph                             = line['graph_query']
                    mod_graph, root, tree_flag        = create_iso_graph(graph)

                    if tree_flag:
                        iso_num                       =f'T-{get_isomorphism_id(mod_graph, root, update=True)}' 
                    else:
                        iso_num                       =f'C-{get_non_tree_isomorphism_id(mod_graph, root, update=True)}'
                except Exception as e:
                    continue


    for idx, iso in enumerate(tree_isomorphisms):
        net = Network()
        # import pdb; pdb.set_trace()
        print(idx, iso['graph'].nodes(data=True), iso['graph'].edges(data=True))
        net.from_nx(iso['graph'])
        net.save_graph(f'../iso_graphs/tree_{idx}.html')
        
        
    for idx, iso in enumerate(non_tree_isomorphisms):
        net = Network()
        # import pdb; pdb.set_trace()
        print(idx, iso['graph'].nodes(data=True), iso['graph'].edges(data=True))
        net.from_nx(iso['graph'])
        net.save_graph(f'../iso_graphs/non_tree_{idx}.html')    


    with open('../output_data/isomorphisms.dill', 'wb') as outfile:
        dill.dump(tree_isomorphisms, outfile)

    with open('../output_data/non-tree-isomorphisms.dill','wb') as outfile:
        dill.dump(non_tree_isomorphisms, outfile)



    # predict the isomorphism for each of the query graph and the predicted s-expression

    for dataset in datasets:
        isomorphism_data = []

        for split in splits[dataset]:

            data                                  = json.load(open(f'../input_data/{dataset}.{split}.json'))

            for line in tqdm(data):
                try:
                    graph                             = line['graph_query']
                    mod_graph, root, tree_flag        = create_iso_graph(graph)

                    if tree_flag:
                        iso_num                       =f'T-{get_isomorphism_id(mod_graph, root, update=False)}' 
                    else:
                        iso_num                       =f'C-{get_non_tree_isomorphism_id(mod_graph, root, update=False)}'


                    s_expression                      = line['s_expression']

                    # obtain the query graph and the networkx graph from the s_expression

                    query_graph, G                    = get_query_graph(s_expression)

                    curr_graph, qnid, tree_iso        = create_iso_graph(query_graph)

                    if tree_iso:
                        pred_iso_num                  = f'T-{get_isomorphism_id(curr_graph, qnid, update=False)}'
                    else:
                        pred_iso_num                  = f'C-{get_non_tree_isomorphism_id(curr_graph, qnid, update=False)}'
                    
                
                    line['isomorphism']               = iso_num
                    line['pred_isomorphism']          = pred_iso_num
                    line['split']                     = split

                    isomorphism_data.append(line)
                except Exception as e:
                    continue

        with open(f'../output_data/{dataset}_isomorphism.json', 'w') as f:
            json.dump(isomorphism_data, f, indent=4)


def get_seen_info(datasets):
    seen_dict            = ddict(lambda: ddict(int))
    seen_dict_by_dataset = ddict(lambda: ddict(lambda: ddict(int)))

    schema_list          = []
    lf_dict              = ddict(list)

    for dataset in datasets:
        tot_data            = [json.load(open(f'../input_data/{dataset}.train.json'))]
        for curr_data in tot_data:
            for elem in tqdm(curr_data):
                graph               = elem['graph_query']

                if graph is not None and graph['nodes'] != []:

                    nodes, edges    = graph['nodes'], graph['edges']
                    
                    ent_list        = [str(node['id']) for node in nodes if node['node_type']!='class']
                    class_list      = [node['class'] for node in nodes]
                    rels_list       = [edge['relation'] for edge in edges]
                    inv_rels_list   = [reverse_properties[edge['relation']] if edge['relation'] in reverse_properties else edge['relation'] for edge in edges]
                    func_list       = [node['function'] for node in nodes]
                    
                    class_func_list = [(node['class'], node['function']) for node in nodes if node['question_node']==0]
                    
                    mod_s_expr      = str(elem['s_expression'])                
                    for ent in ent_list:
                        mod_s_expr  = mod_s_expr.replace(ent, '[ENT]')
                    
                    seen_dict['logical_forms'][mod_s_expr] +=1
                    seen_dict_by_dataset[dataset]['logical_forms'][mod_s_expr] +=1

                    lf_dict[mod_s_expr].append(elem['question'])
                    
                    schema_dict     = {'class':[node['class'] for node in nodes],\
                        'rel':rels_list, 'inv_rel': inv_rels_list, 'func': func_list,\
                        'class_func': class_func_list, 'question': elem['question']}
                    
                    schema_list.append(schema_dict)
                    
                    for cls in class_list:
                        seen_dict['class'][cls] +=1
                        seen_dict_by_dataset[dataset]['class'][cls] +=1
                    
                    for rel in rels_list:
                        seen_dict['rels'][rel] +=1
                        seen_dict_by_dataset[dataset]['rels'][rel] +=1
                    
                    for rel in inv_rels_list:
                        seen_dict['rels'][rel] +=1
                        seen_dict_by_dataset[dataset]['rels'][rel] +=1
                    
                    for func in func_list:
                        seen_dict['funcs'][func]+=1
                        seen_dict_by_dataset[dataset]['funcs'][func] +=1
                
    return seen_dict, seen_dict_by_dataset, lf_dict, schema_list
    


def get_pred_level(graph, s_expr, seen_dict, schema_list, seen_lf_list):

    nodes, edges    = graph['nodes'], graph['edges']
    ent_list        = [str(node['id']) for node in nodes if node['node_type']!='class']
    class_list      = [node['id'] for node in nodes if node['node_type']=='class']
    rels_list       = [edge['relation'] for edge in edges]              
    func_list       = [node['function'] for node in nodes]
    
    new_cls         =  len(set(class_list) - set(list(seen_dict['class'].keys())))
    new_rel         =  len(set(rels_list)  - set(list(seen_dict['rels'].keys())))
    new_func        =  len(set(func_list)  - set(list(seen_dict['funcs'].keys())))

    mod_s_expr                  = s_expr
    for ent in ent_list:
        mod_s_expr              = mod_s_expr.replace(ent, '[ENT]')

    if new_cls != 0 or new_rel !=0 or new_func != 0: 
        level                = 'zero-shot'
    else:
        schema_flag          = False
        em                   = 0
        
        for schema in schema_list:
            schema_rels, schema_inv_rels = schema['rel'], schema['inv_rel']
            pos              = None
            
            if len(rels_list) != len(schema_rels):
                continue
            
            rel_flag    = True                                    
            for rel in rels_list:
                if pos is not None:
                    schema_rels.pop(pos)
                    schema_inv_rels.pop(pos)
                if rel in schema_rels:
                    pos = schema_rels.index(rel)
                elif rel in schema_inv_rels:
                    pos = schema_inv_rels.index(rel)
                else:
                    rel_flag = False
                    break
                                            
            if rel_flag:
                schema_flag = True
                schema_question = schema['question']
                break
        
        if schema_flag:
            for lf in seen_lf_list:
                em = matcher.same_logical_form(mod_s_expr, lf)
                if em       == 1:
                    match_lf = lf
                    break         
                
        if schema_flag and em==1:
            level       = 'i.i.d.'
        else:
            level       = 'compositional'

    return level



def create_dataset_splits():

    # obtain the list of schema items and s-expressions in the training split of the dataset
    datasets                                                = ['webqsp']
    seen_dict, seen_dict_by_dataset, lf_dict, schema_list   = get_seen_info(datasets)
    seen_lf_list                                            = list(set(seen_dict['logical_forms'].keys()))

    for dataset in datasets:
        test_data              = json.load(open(f'../input_data/{dataset}.test.json'))

        level_dict             = {}

        for elem in tqdm(test_data):
            
            level_dict[elem['qid']] = get_pred_level(elem['graph_query'], elem['s_expression'], seen_dict, schema_list, seen_lf_list)
                            
        
        with open(f'../output_data/{dataset}_level.json', 'w') as f:
            json.dump(level_dict, f, indent=4)


# create_dataset_splits()

def conv_sexpr2QG():

    for split in ['train', 'dev', 'test']:
        # read a jsonl file
        iso_data                = []        
        with open(f'/data/shire/projects/GenQA/GNN_codes/codistill-KBQA/data/webqsp/{split}.jsonl') as f:
            for line in f:
                curr_data                         = json.loads(line.strip())
                s_expr                            = curr_data['s_expression']
                query_graph, G                    = get_query_graph(s_expr)
                curr_graph, qnid, tree_iso        = create_iso_graph(query_graph)


                if tree_iso:
                    iso_num                       =f'T-{get_isomorphism_id(curr_graph, qnid, update=False)}'
                
                else:
                    iso_num                       =f'C-{get_non_tree_isomorphism_id(curr_graph, qnid, update=False)}'
                
                curr_data['isomorphism']          = iso_num
                curr_data['tree_iso']             = tree_iso
                
                iso_data.append(curr_data)

                print(iso_num, tree_iso, curr_data['question'], s_expr)

        with open(f'/data/shire/projects/GenQA/GNN_codes/codistill-KBQA/data/webqsp/{split}_isomorphism.json', 'w') as f:
            json.dump(iso_data, f, indent=4)

            

'''

if tree_isomorphisms == []:
    with open('../../output_data/isomorphisms.dill', 'rb') as infile:
        tree_isomorphisms = dill.load(infile)

if non_tree_isomorphisms == []:
    with open('../../output_data/non-tree-isomorphisms.dill', 'rb') as infile:
        non_tree_isomorphisms = dill.load(infile)

'''

if tree_isomorphisms == []:
    with open('isomorphisms.dill', 'rb') as infile:
        tree_isomorphisms = dill.load(infile)

if non_tree_isomorphisms == []:
    with open('non-tree-isomorphisms.dill', 'rb') as infile:
        non_tree_isomorphisms = dill.load(infile)


# conv_sexpr2QG()  # Call the function to convert s-expressions to query graphs



