import os
from pprint import pprint
import argparse
import json

from create_sem_matcher import process_ontology
from create_isomorphisms import *

def create_isomorphism_datasets():

    datasets = ['grailqa', 'webqsp', 'graphquestions', 'cwq']

    splits  = {
                'webqsp': ['train', 'test'],
                'grailqa': ['train', 'test', 'dev'],
                'cwq': ['train', 'test', 'dev'],
                'graphquestions': ['train', 'test']
            }

    # predict the isomorphism for each of the query graph

    mismatches = ddict(lambda: ddict(int))

    for dataset in datasets:
        for split in splits[dataset]:

            data                                        = json.load(open(f'../input_data/{dataset}.{split}.json'))

            iso_data_path                               = f'../output_data/{dataset}.{split}.json'
            iso_data                                    = []

            print(f'Processing {dataset} {split} with {len(data)} queries')

            for line in tqdm(data):
                
                gq_iso_flag, sexpr_iso_flag          = False, False

                try:
                    
                    if 'graph_query' not in line:
                        gq_iso_flag = False

                    if len(line['graph_query']['edges']) ==0:
                        gq_iso_flag = False
                    
                    
                    graph                                   = line['graph_query']
                    mod_graph, root, tree_flag              = create_iso_graph(graph)

                    if tree_flag:
                        gq_iso_num                          =f'T-{get_isomorphism_id(mod_graph, root, update=True)}' 
                    else:
                        gq_iso_num                          =f'C-{get_non_tree_isomorphism_id(mod_graph, root, update=True)}'

                    gq_iso_flag                             = True
                    
                    if 's_expression' in line and line['s_expression'] is not None:

                        sexpr                              = line['s_expression']
                        query_graph, G                     = get_query_graph(sexpr)

                        curr_graph, qnid, tree_iso         = create_iso_graph(query_graph)

                        if tree_iso:
                            sexpr_iso_num                  = f'T-{get_isomorphism_id(curr_graph, qnid, update=False)}'
                        else:
                            sexpr_iso_num                  = f'C-{get_non_tree_isomorphism_id(curr_graph, qnid, update=False)}'
                        
                        sexpr_iso_flag                     = True
                        

                    if gq_iso_flag and sexpr_iso_flag:

                        if sexpr_iso_num != gq_iso_num:
                            mismatches[dataset][split] += 1
                        
                        line['gq_iso_num'] = gq_iso_num
                        line['sexpr_iso_num'] = sexpr_iso_num
                
                    if sexpr_iso_flag and not gq_iso_flag:
                        # print(f'Error: S-expression exists without graph query for {dataset} {split} {line["id"]}')
                        line['gq_iso_num'] = None
                        line['sexpr_iso_num'] = sexpr_iso_num
                    
                    # if gq_iso_flag and not sexpr_iso_flag:
                    #     # print(f'Error: Graph query exists without S-expression for {dataset} {split} {line["id"]}')
                    #     line['gq_iso_num'] = gq_iso_num
                    #     line['sexpr_iso_num'] = None

                    iso_data.append(line)
                    
                except Exception as e:
                    # print(f'Error processing {dataset} {split}: {e}')
                    continue
            
            # report mismatches
            
            print(f'Found {mismatches[dataset][split]} out of {len(data)} mismatches in {dataset} {split}')

            
            # save the isomorphism data
            with open(iso_data_path, 'w') as f:
                json.dump(iso_data, f, indent=4)



def validate_relations():

    ### piece of code to validate whether the set of all relations between a pair of entities includes the relations in the query graph or isomorphism

    datasets = ['grailqa']# 'webqsp', 'graphquestions', 'cwq']

    splits  = {
                # 'webqsp': ['train', 'test'],
                'grailqa': ['train', 'dev'],
                # 'cwq': ['train', 'test', 'dev'],
                # 'graphquestions': ['train', 'test']
            }
    
    fb_roles_file               =  '../../ontology/fb_roles'
    fb_types_file               =  '../../ontology/fb_types'
    reverse_properties_file     =  '../../ontology/reverse_properties'
    domain_dict_file            =  '../../ontology/domain_dict'

    domain_dict                 =  json.load(open(domain_dict_file))

    reverse_properties, relation_dr, relations, upper_types, types, common_rels = process_ontology(fb_roles_file, fb_types_file, reverse_properties_file)

    classes2domains             = {}
    dup_domains                 = 0
    for domain in domain_dict:
        for elem in domain_dict[domain]:
            if elem not in classes2domains:
                classes2domains[elem] = domain
            else:
                dup_domains +=1
    
    print(f'Found {len(classes2domains)} unique classes in domain dict')
    print(f'Found {dup_domains} duplicate classes in domain dict')

    domain_dists                                     = ddict(lambda: ddict(list))
    for dataset in datasets:
        for split in splits[dataset]:

            iso_data_path                            = f'../output_data/{dataset}.{split}.json'
            iso_data                                 = json.load(open(iso_data_path))

            print(f'Validating relations for {dataset} {split} with {len(iso_data)} queries')

            for line in tqdm(iso_data):

                query_graph                         = line['graph_query']
                nodes, edges                        = query_graph['nodes'], query_graph['edges']
                domains                             = set()

                for node in nodes:
                    domains.add(classes2domains[node['class']])                    
                domain_dists[dataset][split].append(len(domains))


    for dataset in domain_dists:
        for split in domain_dists[dataset]:
            avg_dist = sum(domain_dists[dataset][split]) / len(domain_dists[dataset][split])
            print(f'Average domain numbers for {dataset} {split}: {avg_dist:.4f}')


    # relation_dr[fields[1]] = (fields[0], fields[2])

    uptype_dict                     = ddict(set)
    lowtype_dict                    = ddict(set)
    
    for type_ in upper_types:

        if type_.startswith('common.')  or type_.startswith('kg.') or type_.startswith('user.') or type_.startswith('base.') or type_.startswith('freebase.'):
            continue

        up_types = list(upper_types[type_])

        for up_type_ in up_types:
            
            if up_type_.startswith('common.') or up_type_.startswith('kg.') or up_type_.startswith('user.') or up_type_.startswith('base.') or up_type_.startswith('freebase.'):
                continue
            
            uptype_dict[type_].add(up_type_)
            lowtype_dict[up_type_].add(type_)

    print(len(uptype_dict),  'upper types found')
    print(len(lowtype_dict), 'lower types found')

    pairwise_rels               = ddict(list)

    rev_relations               = {}

    for rel in reverse_properties:

        try:
            assert reverse_properties[reverse_properties[rel]] == rel, f'Inconsistent reverse relation for {rel}'
        except Exception as e:
            print(rel, reverse_properties[rel])
            

        rev_relations[reverse_properties[rel]] = rel
        rev_relations[rel] = reverse_properties[rel]

    for rel in relation_dr:

        src, tgt = relation_dr[rel]

        sub_srcs = list(lowtype_dict[src])
        sub_tgts = list(lowtype_dict[tgt])

        srcs     = [src] + sub_srcs
        tgts     = [tgt] + sub_tgts

        # srcs     = [src]
        # tgts     = [tgt]

        for src in srcs:
            for tgt in tgts:
                pairwise_rels[(src, tgt)].append(rel)
                if rel in rev_relations:
                    pairwise_rels[(tgt, src)].append(rev_relations[rel])
                
    print(len(pairwise_rels), 'pairwise relations found')

    #### We now need to validate the relation coverage for each instance of the isomorphism dataset

    relation_coverage       = ddict(lambda: ddict(list))
    subgraph_size           = ddict(lambda: ddict(list))

    for dataset in datasets:
        for split in splits[dataset]:

            iso_data_path                           = f'../output_data/{dataset}.{split}.json'
            iso_data                                 = json.load(open(iso_data_path))

            print(f'Validating relations for {dataset} {split} with {len(iso_data)} queries')

            for line in tqdm(iso_data):

                query_graph     = line['graph_query']
                nodes, edges    = query_graph['nodes'], query_graph['edges']

                relations       = set([edge['relation'] for edge in edges])
                ent_classes     = [node['class'] for node in nodes]

                ent_pairs       = []
                poss_relations  = []

                for idx1, ent1 in enumerate(ent_classes):
                    for idx2, ent2 in enumerate(ent_classes):
                        
                        if idx1 == idx2:
                            continue

                        pair        = (ent1, ent2)
                        
                        ent_pairs.append(pair)

                        if pair in pairwise_rels:
                            poss_relations.extend(pairwise_rels[pair])
                        
                        # opp_pair    = (ent2, ent1)

                        # ent_pairs.append(opp_pair)
                        # if opp_pair in pairwise_rels:
                        #     poss_relations.extend(pairwise_rels[opp_pair])
                        
                        
                ent_pairs      = set(ent_pairs)                        
                poss_relations = set(poss_relations)

                rem_relations  = relations - poss_relations

                coverage       =  (len(relations) - len(rem_relations))/ len(relations) 

                relation_coverage[dataset][split].append(coverage)
                subgraph_size[dataset][split].append(len(poss_relations))

        

        for dataset in relation_coverage:
            for split in relation_coverage[dataset]:
                avg_cov = sum(relation_coverage[dataset][split]) / len(relation_coverage[dataset][split])
                avg_size = sum(subgraph_size[dataset][split]) / len(subgraph_size[dataset][split])
                print(f'Average relation coverage for {dataset} {split}: {avg_cov:.4f}')
                print(f'Average number of relations for {dataset} {split}: {avg_size:.4f}')

    # import pdb; pdb.set_trace()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process isomorphism datasets')
    parser.add_argument('--step', required=True, help='Step to perform')

    args = parser.parse_args()

    if args.step == 'create':
        create_isomorphism_datasets()
    
    if args.step == 'val_rel':
        validate_relations()