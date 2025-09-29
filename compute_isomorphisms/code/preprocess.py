import random
import pandas as pd
import networkx as nx
import numpy as np
import json, os, pickle, sys
from collections import Counter, defaultdict as ddict, OrderedDict as odict
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import argparse
import networkx as nx
from karateclub.node_embedding.neighbourhood import Walklets
import dill
import torch

from create_sem_matcher import process_ontology
# from create_isomorphisms import *

def read_list_file(file_path: str):
    res = []
    if not os.path.isfile(file_path):
        return None
    with open(file_path, 'r', encoding='UTF-8') as file:
        for line in file.readlines():
            res.append(line.strip('\n'))
    return res


def validate_schema_data(schema_data, inp_data, rel2tuples, tuples2rel, rev_relations):

    query_graph     = inp_data['graph_query']
    nodes, edges    = query_graph['nodes'], query_graph['edges']
    relations       = set([edge['relation'] for edge in edges])

    # if split == 'test':
    #     classes              = [node['id'] for node in nodes if node['node_type'] != 'entity' and node['node_type'] != 'literal']
    #     # starting_classes     = [node['id'] for node in nodes if node['node_type'] == 'entity' or node['node_type'] == 'literal']
    # else:

    classes              = [node['class'] for node in nodes]
    starting_classes     = [node['class'] for node in nodes if node['node_type'] == 'entity' or node['node_type'] == 'literal']

    predicted_classes        = schema_data['classes'][:args.top_k]
    predicted_relations      = schema_data['relations'][:args.top_k]

    predicted_classes.extend(starting_classes)

    predicted_classes        = list(set(predicted_classes))
    
    kg_tuples                = []

    for idx1, head in enumerate(predicted_classes):
        for idx2, tail in enumerate(predicted_classes):
            if idx1 == idx2 : continue

            # import pdb; pdb.set_trace()

            curr_rels       = tuples2rel.get((head, tail), [])

            for rel in curr_rels:
                kg_tuples.append((head, rel, tail))
                
            
    for rel in predicted_relations:
        if rel not in rel2tuples: continue
        for head, tail in rel2tuples[rel]:
            kg_tuples.append((head, rel, tail))
        for head, tail in rel2tuples[rev_relations[rel]]:
            kg_tuples.append((head, rel, tail))

    kg_tuples       = list(set(kg_tuples))
    pred_classes    = []
    pred_relations  = []

    for head, rel, tail in kg_tuples:
        pred_classes.append(head)
        pred_classes.append(tail)
        pred_relations.append(rel)

    class_coverage      = 1 - (len(set(classes) - set(pred_classes)) / len(set(classes)))
    relation_coverage   = 1 - (len(set(relations) - set(pred_relations)) / len(set(relations)))
    

    return {'class_coverage': class_coverage, 'relation_coverage': relation_coverage, 'kg_tuples': kg_tuples}

    


def validate_schema_data_grailqapp(schema_data, inp_data, rel2tuples, tuples2rel, rev_relations):

    query_graph     = inp_data['graph_query']
    nodes, edges    = query_graph['nodes'], query_graph['edges']
    relations       = set([edge['relation'] for edge in edges])

    ## since for the test split the classes are missing in the dataset, we need to create a better way to obtain the classes

    all_nodes       = {}
    rem_nids        = {}

    for node in nodes:
        if node['node_type'] == 'class':
            node['class']     = node['id']
            all_nodes[node['nid']] = node
        else:
            rem_nids[node['nid']] = node
    
    possible_classes = ddict(list)
    
    for edge in edges:

        start_nid, end_nid = edge['start'], edge['end']

        if start_nid not in rem_nids and end_nid not in rem_nids: continue

        relation = edge['relation']
        tups     = rel2tuples.get(relation, [])
        
        
        for head, tail in tups:

            if start_nid in rem_nids and end_nid in all_nodes:
                assert tail == all_nodes[end_nid]['class'], f'Inconsistent tail class for edge {edge}'

                possible_classes[start_nid].append(head)

            elif end_nid in rem_nids and start_nid in all_nodes:
                assert head == all_nodes[start_nid]['class'], f'Inconsistent head class for edge {edge}'

                possible_classes[end_nid].append(tail)

            elif start_nid in rem_nids and end_nid in rem_nids:
                possible_classes[start_nid].append(head)
                possible_classes[end_nid].append(tail)
    
    
    for nid in rem_nids:
        if nid not in possible_classes or len(possible_classes[nid]) == 0:
            print(f'No possible classes for node {rem_nids[nid]}')
            # import pdb; pdb.set_trace()
            
        class_counts    = Counter(possible_classes[nid])
        try:
            assigned_class  = class_counts.most_common(1)[0][0]
        except Exception as e:
            assigned_class  = 'type.object'

        rem_nids[nid]['class'] = assigned_class
        all_nodes[nid] = rem_nids[nid]

    # import pdb; pdb.set_trace()
    nodes                       = list(all_nodes.values())
    # if split == 'test':
    #     classes              = [node['id'] for node in nodes if node['node_type'] != 'entity' and node['node_type'] != 'literal']
    #     # starting_classes     = [node['id'] for node in nodes if node['node_type'] == 'entity' or node['node_type'] == 'literal']
    # else:

    classes              = [node['class'] for node in nodes]
    starting_classes     = [node['class'] for node in nodes if node['node_type'] == 'entity' or node['node_type'] == 'literal']

    predicted_classes        = schema_data['classes'][:args.top_k]
    predicted_relations      = schema_data['relations'][:args.top_k]

    predicted_classes.extend(starting_classes)

    predicted_classes        = list(set(predicted_classes))
    
    kg_tuples                = []

    for idx1, head in enumerate(predicted_classes):
        for idx2, tail in enumerate(predicted_classes):
            if idx1 == idx2 : continue

            # import pdb; pdb.set_trace()

            curr_rels       = tuples2rel.get((head, tail), [])

            for rel in curr_rels:
                kg_tuples.append((head, rel, tail))
                
            
    for rel in predicted_relations:
        if rel not in rel2tuples: continue
        for head, tail in rel2tuples[rel]:
            kg_tuples.append((head, rel, tail))
        for head, tail in rel2tuples[rev_relations[rel]]:
            kg_tuples.append((head, rel, tail))

    kg_tuples       = list(set(kg_tuples))
    pred_classes    = []
    pred_relations  = []

    for head, rel, tail in kg_tuples:
        pred_classes.append(head)
        pred_classes.append(tail)
        pred_relations.append(rel)

    class_coverage      = 1 - (len(set(classes) - set(pred_classes)) / len(set(classes)))
    relation_coverage   = 1 - (len(set(relations) - set(pred_relations)) / len(set(relations)))
    

    return {'class_coverage': class_coverage, 'relation_coverage': relation_coverage, 'kg_tuples': kg_tuples}

    



def create_grailqa():

    relations_dict                  =    {}
    relations_dict['self_loop']     =    len(relations_dict)

    input_dir                       =    '/data/shire/projects/GenQA/ISOKBQA/KC/papers/TIARA/dataset/GrailQA/'


    class_list                      =    read_list_file(f'{input_dir}/grail_classes.txt')
    relations_list                  =    read_list_file(f'{input_dir}/grail_relations.txt')

    ### initializing the relations dictionary with the relations from the ontology
    for rel in relations_list:
        relations_dict[rel]         =    len(relations_dict)

    print(f'Loaded {len(relations_list)} relations from the relations list file')
    
    class_set                       =    set(class_list)   

    ### loading the ontology files #####
    fb_roles_file                   =  '../../ontology/fb_roles'
    fb_types_file                   =  '../../ontology/fb_types'
    reverse_properties_file         =  '../../ontology/reverse_properties'
    domain_dict_file                =  '../../ontology/domain_dict'

    fb_tuples_file                  =  '../../ontology/fb_tuples'

    fb_tuples_list                  =  []

    rel2tuples_dict                 =  ddict(list)
    tuples2rel_dict                 =  ddict(list)

    domain_dict                     =  json.load(open(domain_dict_file))

    reverse_properties, relation_dr, relations, upper_types, types, common_rels = process_ontology(fb_roles_file, fb_types_file, reverse_properties_file)

    rev_relations                   = {}

    for rel in reverse_properties:
        try:
            assert reverse_properties[reverse_properties[rel]] == rel, f'Inconsistent inverse relation for {rel}'
        except Exception as e:
            print(rel, reverse_properties[rel])
            
        rev_relations[reverse_properties[rel]] = rel
        rev_relations[rel] = reverse_properties[rel]
    
    print(f'Original number of reverse relations: {len(rev_relations)}')

    ''''
    Actually making sure the relations list are unique
    '''

    ### adding inverse relations to the relations dictionary for the ones which are not already present ###
    for rel in relations_list:
        if rel == 'self_loop':
            continue
        if rel not in rev_relations:
            rev_relations[rel]          = f'rev_{rel}'
            rev_relations[f'rev_{rel}'] = rel
    
    print(f'Updated number of reverse relations: {len(rev_relations)}')
    
    for rel in rev_relations:
        if rel not in relations_dict:
            relations_dict[rel]                 = len(relations_dict)
            # relations_dict[rev_relations[rel]]  = len(relations_dict)

    print(f'Total number of relations (including inverse relations): {len(relations_dict)}')

    rel2id = {}
    id2rel = {}

    for rel in relations_dict:
        rel2id[rel] = relations_dict[rel]
        id2rel[relations_dict[rel]] = rel
    
    print(f'Length of rel2id dictionary: {len(rel2id)}')
    print(f'Length of id2rel dictionary: {len(id2rel)}')

    # import pdb; pdb.set_trace()

    for line in open(fb_tuples_file, 'r'):
        head, rel, tail                = line.strip().split('\t')
        if rel in relations_dict and head in class_set and tail in class_set:
            
            # import pdb; pdb.set_trace()
            fb_tuples_list.append((head, rel, tail))
            fb_tuples_list.append((tail, rev_relations[rel], head))

            rel2tuples_dict[rel].append((head, tail))
            rel2tuples_dict[rev_relations[rel]].append((tail, head))
            tuples2rel_dict[(head, tail)].append(rel)
            tuples2rel_dict[(tail, head)].append(rev_relations[rel])

            # fb_tuples_dict[rel].append((head, tail))
            # fb_tuples_dict[rev_relations[rel]].append((tail, head))
    fb_tuples_list = list(set(fb_tuples_list))

    print(f'Loaded {len(fb_tuples_list)} tuples from the ontology')
    print(f'Loaded {len(rel2tuples_dict)} relations with {len(tuples2rel_dict)} tuples')
    print(f'Loaded {len(relations_dict)} relations in total')

    with open(f'{args.data_dir}/{args.dataset}/relations_dict.json', 'w') as f:
        json.dump(relations_dict, f, indent=4)
    
    with open(f'{args.data_dir}/{args.dataset}/fb_tuples_list.json', 'w') as f:
        json.dump(fb_tuples_list, f, indent=4)
    

    subgraph_dict = ddict(lambda: ddict(list))    
    
    KBQA_dict                                              = {'train': [], 'dev': [], 'test': []}

    statistics_dict                                        = ddict(lambda: ddict(list))

    for split in ['train', 'dev']:

        relation_coverage_dict                             = ddict(list)

        filtered_input_data                                = json.load(open(f'../output_data/grailqa.{split}.filtered.json'))

        if split in ['dev', 'test']:
            filtered_schema_data                           = f'/data/shire/projects/GenQA/ISOKBQA/KC/papers/TIARA/dataset/GrailQA/schema_linking_results/dense_retrieval_{args.dataset}_{split}-filtered.jsonl'
        else:
            filtered_schema_data                           = f'/data/shire/projects/GenQA/ISOKBQA/KC/papers/TIARA/dataset/GrailQA/schema_linking_results/dense_retrieval_{args.dataset}_{split}.jsonl'

        filtered_input_dict                                = {}

        for elem in filtered_input_data:
            filtered_input_dict[str(elem['qid'])]          = elem

        filtered_schema_dict                               = {}
        filtered_schema_data                               = [json.loads(line) for line in open(filtered_schema_data, 'r')]

        for elem in filtered_schema_data:
            filtered_schema_dict[str(elem['qid'])]         = elem

        print(f'Processing {args.dataset} {split} with {len(filtered_input_data)} queries and {len(filtered_schema_data)} schema links')


        for qid in tqdm(filtered_input_dict, desc=f'Creating KBQA data for {split}'):

            try:

                curr_data                                    = {}
                inp_data                                     = filtered_input_dict[qid]
                schema_data                                  = filtered_schema_dict.get(qid, None)

                if schema_data is None:
                    print(f'No schema data for {qid}, skipping...')
                    continue

                assert str(inp_data['qid']) == str(schema_data['qid']), f'QID mismatch for {inp_data["qid"]}'

                kg_data         = validate_schema_data(schema_data, inp_data, rel2tuples_dict, tuples2rel_dict, rev_relations)

                if kg_data is None: continue

                kg_tuples       = kg_data['kg_tuples']
                class_coverage  = kg_data['class_coverage']
                relation_coverage = kg_data['relation_coverage']

                statistics_dict[split]['class_coverage'].append(class_coverage)
                statistics_dict[split]['relation_coverage'].append(relation_coverage)
                statistics_dict[split]['num_kg_tuples'].append(len(kg_tuples))

                if class_coverage <1 or relation_coverage < 1: continue            

                # curr_data       = json.loads(line.strip())
                # kg_tuples       = curr_data['kg_tuples']

                if len(kg_tuples) == 0: continue

                

                curr_data['ID']            = inp_data['qid']
                curr_data['question']      = inp_data['question']
                curr_data['answers']       = []


                for ans in inp_data['answer']:
                    if ans['answer_type'] == 'Entity':
                        curr_data['answers'].append([ans['answer_argument'], ans['entity_name']])                    
                    if ans['answer_type'] == 'Value':
                        curr_data['answers'].append([ans['answer_argument'], ans['answer_argument']])
        

                curr_data['s_expression']  = inp_data['s_expression']

                if split == 'train':
                    curr_data['level']     = 'i.i.d.'
                else:
                    curr_data['level']     = inp_data['level']    

                curr_data['isomorphism']   = inp_data['gq_iso_num']
                curr_data['kg_tuples']     = kg_tuples


                head_ents                       = []
                head_classes                    = []
                ans_classes                     = []

                for node in inp_data['graph_query']['nodes']:

                    if node['node_type'] !='class':
                        head_ents.append((node['id'], node['class'], node['friendly_name']))
                        head_classes.append(node['class'])
                    
                    if node['question_node'] == 1:
                        ans_classes.append(node['class'])
                
                curr_data['graph_query']    = inp_data['graph_query']
                curr_data['head_entities']  = head_ents

                
                head_classes                = set(head_classes)
                ans_classes                 = set(ans_classes)
                nodes, edges                = set(), set()
                node_dict                   = {}

                for node in list(head_classes):
                    if node not in node_dict:
                        node_dict[node] = len(node_dict)
                    nodes.add(node_dict[node])
                
                for node in list(ans_classes):
                    if node not in node_dict:
                        node_dict[node] = len(node_dict)
                    nodes.add(node_dict[node])

                x, edge_index, edge_type, answer_x, head_x = [], [[],[]], [], [], []

                for head, rel, tail in kg_tuples:
                    if head not in node_dict:
                        node_dict[head] = len(node_dict)

                    if tail not in node_dict:
                        node_dict[tail] = len(node_dict)

                    nodes.add(node_dict[head])
                    nodes.add(node_dict[tail])

                    edges.add((node_dict[head], node_dict[tail]))
                    edges.add((node_dict[tail], node_dict[head]))

                    edge_index[0].append(node_dict[head])
                    edge_index[1].append(node_dict[tail])
                    edge_index[0].append(node_dict[tail])
                    edge_index[1].append(node_dict[head])
                    edge_type.append(relations_dict[rel])

                    assert rel in rev_relations, f'Relation {rel} not in reverse relations'

                    rev_rel = rev_relations[rel]
                    edge_type.append(relations_dict[rev_rel])  

                for node in node_dict:
                    edge_index[0].append(node_dict[node])
                    edge_index[1].append(node_dict[node])
                    edge_type.append(relations_dict['self_loop'])

                G = nx.DiGraph()
                G.add_nodes_from(nodes)
                G.add_edges_from(edges)

                model = Walklets(dimensions=192)
                model.fit(G)
                embedding = model.get_embedding()

                # import pdb; pdb.set_trace()

                ans_nodes  = set([node_dict[ent] for ent in list(ans_classes) if ent in node_dict ])
                head_nodes = set([node_dict[ent] for ent in list(head_classes) if ent in node_dict])


                for idx, emb in enumerate(embedding):
                    x.append(emb)
                    if idx in ans_nodes:
                        answer_x.append(1)
                    else:
                        answer_x.append(0)
                    
                    if idx in head_nodes:
                        head_x.append(1)
                    else:
                        head_x.append(0)
                
                x, edge_index, edge_type, answer_x, head_x         = torch.tensor(np.array(x)), torch.LongTensor(edge_index), torch.LongTensor(edge_type), torch.LongTensor(answer_x),torch.LongTensor(head_x)

                node_data = {'x': x, 'edge_index': edge_index, 'edge_type': edge_type, 'answer_x': answer_x, 'head_x': head_x, 'node_dict': node_dict}

                curr_data['node_data'] = node_data
                

                KBQA_dict[split].append(curr_data)
            
            except Exception as e:
                print(f'Error processing {qid}: {e}')
                continue


    with open(f'{args.data_dir}/{args.dataset}/KBQA_walklet_edge_label_768_dim_data.dill', 'wb') as f:
        dill.dump(KBQA_dict, f)
    

    for split in ['train', 'dev']:
        print(f'class_coverage: {np.mean(statistics_dict[split]["class_coverage"])}')
        print(f'relation_coverage: {np.mean(statistics_dict[split]["relation_coverage"])}')
        print(f'num_kg_tuples: {np.mean(statistics_dict[split]["num_kg_tuples"])}')
    

def filter_grailqa():

    with open(f'{args.data_dir}/grailqa/KBQA_walklet_edge_label_768_dim_data.dill', 'rb') as f:
        KBQA_dict = dill.load(f)
    
    new_KBQA_dict = {'train': [], 'dev': []}

    for split in ['train', 'dev']:
        for elem in KBQA_dict[split]:

            num_edges = elem['node_data']['edge_index'].shape[1]

            if num_edges <= 200:
                new_KBQA_dict[split].append(elem)
    
    with open(f'{args.data_dir}/grailqa/KBQA_walklet_edge_label_768_dim_data_filtered.dill', 'wb') as f:
        dill.dump(new_KBQA_dict, f)


def create_grailqapp():

    relations_dict                  =    {}
    relations_dict['self_loop']     =    len(relations_dict)

    input_dir                       =    '/data/shire/projects/GenQA/ISOKBQA/KC/papers/TIARA/dataset/GrailQA/'


    class_list                      =    read_list_file(f'{input_dir}/grail_classes.txt')
    relations_list                  =    read_list_file(f'{input_dir}/grail_relations.txt')

    ### initializing the relations dictionary with the relations from the ontology
    for rel in relations_list:
        relations_dict[rel]         =    len(relations_dict)

    print(f'Loaded {len(relations_list)} relations from the relations list file')
    
    class_set                       =    set(class_list)   

    ### loading the ontology files #####
    fb_roles_file                   =  '../../ontology/fb_roles'
    fb_types_file                   =  '../../ontology/fb_types'
    reverse_properties_file         =  '../../ontology/reverse_properties'
    domain_dict_file                =  '../../ontology/domain_dict'

    fb_tuples_file                  =  '../../ontology/fb_tuples'

    fb_tuples_list                  =  []

    rel2tuples_dict                 =  ddict(list)
    tuples2rel_dict                 =  ddict(list)

    domain_dict                     =  json.load(open(domain_dict_file))

    reverse_properties, relation_dr, relations, upper_types, types, common_rels = process_ontology(fb_roles_file, fb_types_file, reverse_properties_file)

    rev_relations                   = {}

    for rel in reverse_properties:
        try:
            assert reverse_properties[reverse_properties[rel]] == rel, f'Inconsistent inverse relation for {rel}'
        except Exception as e:
            print(rel, reverse_properties[rel])
            
        rev_relations[reverse_properties[rel]] = rel
        rev_relations[rel] = reverse_properties[rel]
    
    print(f'Original number of reverse relations: {len(rev_relations)}')

    ''''
    Actually making sure the relations list are unique
    '''

    ### adding inverse relations to the relations dictionary for the ones which are not already present ###
    for rel in relations_list:
        if rel == 'self_loop':
            continue
        if rel not in rev_relations:
            rev_relations[rel]          = f'rev_{rel}'
            rev_relations[f'rev_{rel}'] = rel
    
    print(f'Updated number of reverse relations: {len(rev_relations)}')
    
    for rel in rev_relations:
        if rel not in relations_dict:
            relations_dict[rel]                 = len(relations_dict)
            # relations_dict[rev_relations[rel]]  = len(relations_dict)

    print(f'Total number of relations (including inverse relations): {len(relations_dict)}')

    rel2id = {}
    id2rel = {}

    for rel in relations_dict:
        rel2id[rel] = relations_dict[rel]
        id2rel[relations_dict[rel]] = rel
    
    print(f'Length of rel2id dictionary: {len(rel2id)}')
    print(f'Length of id2rel dictionary: {len(id2rel)}')

    for line in open(fb_tuples_file, 'r'):
        head, rel, tail                = line.strip().split('\t')
        if rel in relations_dict and head in class_set and tail in class_set:
            
            # import pdb; pdb.set_trace()
            fb_tuples_list.append((head, rel, tail))
            fb_tuples_list.append((tail, rev_relations[rel], head))

            rel2tuples_dict[rel].append((head, tail))
            rel2tuples_dict[rev_relations[rel]].append((tail, head))
            tuples2rel_dict[(head, tail)].append(rel)
            tuples2rel_dict[(tail, head)].append(rev_relations[rel])

            # fb_tuples_dict[rel].append((head, tail))
            # fb_tuples_dict[rev_relations[rel]].append((tail, head))
    fb_tuples_list = list(set(fb_tuples_list))

    print(f'Loaded {len(fb_tuples_list)} tuples from the ontology')
    print(f'Loaded {len(rel2tuples_dict)} relations with {len(tuples2rel_dict)} tuples')
    print(f'Loaded {len(relations_dict)} relations in total')

    # with open(f'{args.data_dir}/{args.dataset}/relations_dict.json', 'w') as f:
    #     json.dump(relations_dict, f, indent=4)
    
    # with open(f'{args.data_dir}/{args.dataset}/fb_tuples_list.json', 'w') as f:
    #     json.dump(fb_tuples_list, f, indent=4)
    
    subgraph_dict = ddict(lambda: ddict(list))    
    
    KBQA_dict                                              = {'train': [], 'dev': [], 'test': []}

    statistics_dict                                        = ddict(lambda: ddict(list))

    bad_cases                                              = ddict(list)

    for split in ['test']:

        relation_coverage_dict                             = ddict(list)

        filtered_input_data                                = json.load(open(f'../output_data/grailqa.{split}.filtered.json'))

        filtered_schema_data                               = f'/data/shire/projects/GenQA/ISOKBQA/KC/papers/TIARA/dataset/GrailQA/schema_linking_results/dense_retrieval_grailqa_test-filtered.jsonl'
        
        filtered_input_dict                                = {}

        for elem in filtered_input_data:
            filtered_input_dict[str(elem['qid'])]          = elem

        filtered_schema_dict                               = {}
        filtered_schema_data                               = [json.loads(line) for line in open(filtered_schema_data, 'r')]

        for elem in filtered_schema_data:
            filtered_schema_dict[str(elem['qid'])]         = elem

        print(f'Processing {args.dataset} {split} with {len(filtered_input_data)} queries and {len(filtered_schema_data)} schema links')


        for qid in tqdm(filtered_input_dict, desc=f'Creating KBQA data for {split}'):

            curr_data                                    = {}
            inp_data                                     = filtered_input_dict[qid]
            schema_data                                  = filtered_schema_dict.get(qid, None)

            if schema_data is None:
                print(f'No schema data for {qid}, skipping...')
                continue

            assert str(inp_data['qid']) == str(schema_data['qid']), f'QID mismatch for {inp_data["qid"]}'

            kg_data         = validate_schema_data_grailqapp(schema_data, inp_data, rel2tuples_dict, tuples2rel_dict, rev_relations)
            

            if kg_data is None: 

                bad_cases['no_schema_data'].append(qid)
                
                continue

            kg_tuples       = kg_data['kg_tuples']
            class_coverage  = kg_data['class_coverage']
            relation_coverage = kg_data['relation_coverage']

            statistics_dict[split]['class_coverage'].append(class_coverage)
            statistics_dict[split]['relation_coverage'].append(relation_coverage)
            statistics_dict[split]['num_kg_tuples'].append(len(kg_tuples))

            # import pdb; pdb.set_trace()
            if class_coverage <1 or relation_coverage < 1: 
                bad_cases['low_coverage'].append(qid)
                print(f'Low coverage for {qid}: class_coverage={class_coverage}, relation_coverage={relation_coverage}')
                



            # curr_data       = json.loads(line.strip())
            # kg_tuples       = curr_data['kg_tuples']

            if len(kg_tuples) == 0: 
                bad_cases['no_kg_tuples'].append(qid)
                
                continue

            # import pdb; pdb.set_trace()

            curr_data['ID']            = inp_data['qid']
            curr_data['question']      = inp_data['question']
            curr_data['answers']       = []


            for ans in inp_data['answer']:
                if ans['answer_type'] == 'Entity':
                    curr_data['answers'].append([ans['answer_argument'], ans['entity_name']])                    
                if ans['answer_type'] == 'Value':
                    curr_data['answers'].append([ans['answer_argument'], ans['answer_argument']])

            

            curr_data['s_expression']  = inp_data['s_expression']

            curr_data['level']         = inp_data['pred_level']

            curr_data['isomorphism']   = inp_data['gq_iso_num']
            curr_data['kg_tuples']     = kg_tuples


            head_ents                       = []
            head_classes                    = []
            ans_classes                     = []

            for node in inp_data['graph_query']['nodes']:

                if node['node_type'] !='class':
                    head_ents.append((node['id'], node['class'], node['friendly_name']))
                    head_classes.append(node['class'])
                
                if node['question_node'] == 1:
                    ans_classes.append(node['class'])
            
            curr_data['graph_query']    = inp_data['graph_query']
            curr_data['head_entities']  = head_ents

            
            head_classes                = set(head_classes)
            ans_classes                 = set(ans_classes)
            nodes, edges                = set(), set()
            node_dict                   = {}

            for node in list(head_classes):
                if node not in node_dict:
                    node_dict[node] = len(node_dict)
                nodes.add(node_dict[node])
            
            for node in list(ans_classes):
                if node not in node_dict:
                    node_dict[node] = len(node_dict)
                nodes.add(node_dict[node])

            x, edge_index, edge_type, answer_x, head_x = [], [[],[]], [], [], []

            for head, rel, tail in kg_tuples:
                if head not in node_dict:
                    node_dict[head] = len(node_dict)

                if tail not in node_dict:
                    node_dict[tail] = len(node_dict)

                nodes.add(node_dict[head])
                nodes.add(node_dict[tail])

                edges.add((node_dict[head], node_dict[tail]))
                edges.add((node_dict[tail], node_dict[head]))

                edge_index[0].append(node_dict[head])
                edge_index[1].append(node_dict[tail])
                edge_index[0].append(node_dict[tail])
                edge_index[1].append(node_dict[head])
                edge_type.append(relations_dict[rel])

                assert rel in rev_relations, f'Relation {rel} not in reverse relations'

                rev_rel = rev_relations[rel]
                edge_type.append(relations_dict[rev_rel])  

            for node in node_dict:
                edge_index[0].append(node_dict[node])
                edge_index[1].append(node_dict[node])
                edge_type.append(relations_dict['self_loop'])

            G = nx.DiGraph()
            G.add_nodes_from(nodes)
            G.add_edges_from(edges)

            model = Walklets(dimensions=192)
            model.fit(G)
            embedding = model.get_embedding()

            # import pdb; pdb.set_trace()

            ans_nodes  = set([node_dict[ent] for ent in list(ans_classes) if ent in node_dict ])
            head_nodes = set([node_dict[ent] for ent in list(head_classes) if ent in node_dict])


            for idx, emb in enumerate(embedding):
                x.append(emb)
                if idx in ans_nodes:
                    answer_x.append(1)
                else:
                    answer_x.append(0)
                
                if idx in head_nodes:
                    head_x.append(1)
                else:
                    head_x.append(0)
            
            x, edge_index, edge_type, answer_x, head_x         = torch.tensor(np.array(x)), torch.LongTensor(edge_index), torch.LongTensor(edge_type), torch.LongTensor(answer_x),torch.LongTensor(head_x)

            node_data = {'x': x, 'edge_index': edge_index, 'edge_type': edge_type, 'answer_x': answer_x, 'head_x': head_x, 'node_dict': node_dict}

            curr_data['node_data'] = node_data
            

            KBQA_dict[split].append(curr_data)
        
    with open(f'{args.data_dir}/grailqa/KBQA_walklet_edge_label_768_dim_data_test.dill', 'wb') as f:
        dill.dump(KBQA_dict, f)
    
    print(f'Bad cases for grailqapp test split:')
    print(f'Number of cases with low coverage: {len(bad_cases["low_coverage"])}')
    print(f'Number of cases with no kg_tuples: {len(bad_cases["no_kg_tuples"])}')
    print(f'Number of cases with no schema data: {len(bad_cases["no_schema_data"])}')

    

    for split in ['test']:
        print(f'class_coverage: {np.mean(statistics_dict[split]["class_coverage"])}')
        print(f'relation_coverage: {np.mean(statistics_dict[split]["relation_coverage"])}')
        print(f'num_kg_tuples: {np.mean(statistics_dict[split]["num_kg_tuples"])}')
        print(f'Total number of processed queries: {len(KBQA_dict[split])}')



def stats_dataset():

    with open(f'{args.data_dir}/{args.dataset}/KBQA_walklet_edge_label_768_dim_data.dill', 'rb') as f:
        KBQA_dict = dill.load(f)
    
    stats_dict = ddict(lambda: ddict(list))

    for split in ['train', 'dev', 'test']:

        for elem in KBQA_dict[split]:
            stats_dict[split]['num_nodes'].append(elem['node_data']['x'].shape[0])
            stats_dict[split]['num_edges'].append(elem['node_data']['edge_index'].shape[1])
            stats_dict[split]['num_answers'].append(elem['node_data']['answer_x'].sum().item())
            stats_dict[split]['num_kg_tuples'].append(len(elem['kg_tuples']))

    for split in ['train', 'dev']:
        print(f'Stats for {split}:')
        print(f'Average number of nodes: {np.mean(stats_dict[split]["num_nodes"])}')
        print(f'Average number of edges: {np.mean(stats_dict[split]["num_edges"])}')
        print(f'Average number of answers: {np.mean(stats_dict[split]["num_answers"])}')
        print(f'Average number of kg_tuples: {np.mean(stats_dict[split]["num_kg_tuples"])}')

        print('Max values:')
        print(f'Max number of nodes: {np.max(stats_dict[split]["num_nodes"])}')
        print(f'Max number of edges: {np.max(stats_dict[split]["num_edges"])}')
        print(f'Max number of answers: {np.max(stats_dict[split]["num_answers"])}')
        print(f'Max number of kg_tuples: {np.max(stats_dict[split]["num_kg_tuples"])}')

        print('Distribution of number of nodes:')
        print(Counter(sorted(stats_dict[split]["num_nodes"])))
        print('Distribution of number of edges:')
        print(Counter(sorted(stats_dict[split]["num_edges"])))
        print('Distribution of number of answers:')
        print(Counter(stats_dict[split]["num_answers"]))
        print('Distribution of number of kg_tuples:')
        print(Counter(sorted(stats_dict[split]["num_kg_tuples"])))
    

def sanity_check():

    with open(f'{args.data_dir}/{args.dataset}/KBQA_walklet_edge_label_768_dim_data.dill', 'rb') as f:
        KBQA_dict = dill.load(f)
    
    relations_dict = json.load(open(f'{args.data_dir}/{args.dataset}/relations_dict.json', 'r'))

    print(f'Loaded {len(relations_dict)}')

    bad_counts = {'train': 0, 'dev': 0}
    for split in ['train', 'dev']:
        for elem in tqdm(KBQA_dict[split]):

            kg_tuples = elem['kg_tuples']
            edge_type = elem['node_data']['edge_type']

            for kg_head, kg_rel, kg_tail in kg_tuples:
                if kg_rel not in relations_dict:
                    print(f'Error: Relation {kg_rel} not found in relations_dict for {elem["ID"]}')
                    bad_counts[split] += 1
                    # import pdb; pdb.set_trace()
            
            edge_type_set = set(edge_type.tolist())
            max_edge_number = max(edge_type_set)

            try:
                assert max_edge_number < len(relations_dict)
            except Exception as e:
                # import pdb; pdb.set_trace()
                print(f'Max edge number {max_edge_number} is greater than the number of relations {len(relations_dict)} for {elem["ID"]}')
                bad_counts[split] += 1


    
        print(f'Bad counts for {args.dataset} {split}: {bad_counts[split]}')



                
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../../isomorphism_pred_KBQA/data')
    parser.add_argument('--dataset',  type=str, default='grailqa')
    parser.add_argument('--step',     type=str, required=True)
    parser.add_argument('--top_k',    type=int, default=10, help='Top-K classes and relations to consider for schema linking')


    args    = parser.parse_args()

    
    if args.step == 'create':
        if args.dataset == 'grailqa':
            create_grailqa()
        
        elif args.dataset == 'grailqapp':
            create_grailqapp()
    
    if args.step == 'stats':
        stats_dataset()

    if args.step == 'filter':
        if args.dataset == 'grailqa':
            filter_grailqa()

    if args.step == 'sanity':
        sanity_check()
    

    if args.step == 'poss_class':

        if args.dataset == 'grailqapp':

            choose_classes()
    

    #     elif args.dataset == 'cwq':
    #         create_cwq_KBQA()
        

    # elif args.step == 'sanity':
    #     do_sanity_check()

    # elif args.step == 'add_edge_label':
    #     add_edge_label()

