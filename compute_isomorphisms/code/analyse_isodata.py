import os
from pprint import pprint
import argparse
import json
import networkx as nx
import numpy as np

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

    all_tuples_file             =  open('../../ontology/fb_tuples_all','w')

    domain_dict                 =  json.load(open(domain_dict_file))

    reverse_properties, relation_dr, relations, upper_types, types, common_rels = process_ontology(fb_roles_file, fb_types_file, reverse_properties_file)

    rev_relations               = {}

    for rel in reverse_properties:
        try:
            assert reverse_properties[reverse_properties[rel]] == rel, f'Inconsistent reverse relation for {rel}'
        except Exception as e:
            print(rel, reverse_properties[rel])
            
        rev_relations[reverse_properties[rel]] = rel
        rev_relations[rel] = reverse_properties[rel]

    print(f'Loaded {len(reverse_properties)} relations and {len(rev_relations)} reverse relations')

    # import pdb; pdb.set_trace()

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

        if type_.startswith('common.')  or type_.startswith('kg.') or type_.startswith('user.') or type_.startswith('base.') or type_.startswith('freebase.')  or type_.startswith('fb.'):
            continue
        
        try:
            type2dom = classes2domains[type_]
        except KeyError as e:
            type2dom = type_.split('.')[0]

        up_types  = list(upper_types[type_])

        for up_type_ in up_types:
            
            if up_type_.startswith('common.') or up_type_.startswith('kg.') or up_type_.startswith('user.') or up_type_.startswith('base.') or up_type_.startswith('freebase.') or up_type_.startswith('fb.'):
                continue
            
            try:
                uptype2dom = classes2domains[up_type_]
            except KeyError as e:
                uptype2dom = up_type_.split('.')[0]
            
            # if uptype2dom != type2dom:
            #     continue

            uptype_dict[type_].add(up_type_)
            lowtype_dict[up_type_].add(type_)

    print(len(uptype_dict),  'upper types found')
    print(len(lowtype_dict), 'lower types found')

    pairwise_rels            = ddict(list)
    unique_tuples            = set()

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
                unique_tuples.add((src, tgt, rel))
                

                if rel in rev_relations:
                    pairwise_rels[(tgt, src)].append(rev_relations[rel])
                    unique_tuples.add((tgt, src, rev_relations[rel]))
                
                else:
                    pairwise_rels[(tgt, src)].append(f'rev_{rel}')
                    unique_tuples.add((tgt, src, f'rev_{rel}'))

                
    print(len(pairwise_rels), 'pairwise relations found')
    print(len(unique_tuples), 'unique tuples found')

    for elem in unique_tuples:
        all_tuples_file.write(f'{elem[0]}\t{elem[2]}\t{elem[1]}\n')
    
    all_tuples_file.close()


    #### We now need to validate the relation coverage for each instance of the isomorphism dataset

    relation_coverage       = ddict(lambda: ddict(list))
    subgraph_size           = ddict(lambda: ddict(list))

    diff_scenarios          = ddict(lambda: ddict(list))


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

                # if coverage < 1:
                    # import pdb; pdb.set_trace()

                relation_coverage[dataset][split].append(coverage)
                subgraph_size[dataset][split].append(len(poss_relations))

                function        = line.get('function', None)
                only_ents       = 1 if len(set([node['node_type'] for node in nodes])) == 2 else 2
                isomorphism     = line.get('gq_iso_num', None)

                diff_scenarios[f'{dataset}-{split}'][f'Function:{function}'].append(coverage)
                diff_scenarios[f'{dataset}-{split}'][f'Entities:{only_ents}'].append(coverage)
                diff_scenarios[f'{dataset}-{split}'][f'Isomorphism:{isomorphism}'].append(coverage)

        

        for dataset in relation_coverage:
            for split in relation_coverage[dataset]:
                avg_cov = sum(relation_coverage[dataset][split]) / len(relation_coverage[dataset][split])
                avg_size = sum(subgraph_size[dataset][split]) / len(subgraph_size[dataset][split])
                print(f'Average relation coverage for {dataset} {split}: {avg_cov:.4f}')
                print(f'Average number of relations for {dataset} {split}: {avg_size:.4f}')
            


        for dataset in diff_scenarios:
            for scenario in sorted(diff_scenarios[dataset]):
                avg_cov = sum(diff_scenarios[dataset][scenario]) / len(diff_scenarios[dataset][scenario])
                print(f'Average relation coverage for {dataset} {scenario}: {avg_cov:.4f}')
            print()




def filter_grailqa_isomorphisms():

    datasets = ['grailqa']# 'webqsp', 'graphquestions', 'cwq']

    splits  = {
            'grailqa': ['train', 'dev', 'test'], 
            }
    
    isomorphism_dict = ddict(lambda: ddict(lambda: ddict(int)))

    valid_isomorphisms = ['T-0', 'T-1', 'T-2', 'T-3', 'T-4','T-5']

    for dataset in datasets:
        for split in splits[dataset]:

            iso_data_path                                  = f'../output_data/{dataset}.{split}.json'
            iso_data                                       = json.load(open(iso_data_path))

            filtered_iso_data                              = []
            filtered_iso_data_path                         = f'../output_data/{dataset}.{split}.filtered.json'

            question_only_data                             = []
            question_only_data_path                        = f'../iso_data/{dataset}.{split}.question_only.json'


            print(f'Filtering isomorphisms for {dataset} {split} with {len(iso_data)} queries')

            filtered_iso_data                             = []

            for line in tqdm(iso_data):

                gq_iso_num      = line.get('gq_iso_num', None)
                sexpr_iso_num   = line.get('sexpr_iso_num', None)

                if gq_iso_num  == sexpr_iso_num:

                    if gq_iso_num in valid_isomorphisms:

                        isomorphism_dict[dataset][split][gq_iso_num] += 1
                        isomorphism_dict[dataset][split]['total'] += 1

                        filtered_iso_data.append(line)
            
            with open(filtered_iso_data_path, 'w') as f:
                json.dump(filtered_iso_data, f, indent=4)


            qonly_data          = []
            for elem in filtered_iso_data:
                curr_dict      = {}

                curr_dict['qid']            = elem['qid']
                curr_dict['question']       = elem['question']
                curr_dict['s_expression']   = elem['s_expression']
                curr_dict['isomorphism']    = elem['sexpr_iso_num']

                if 'level' in elem:
                    curr_dict['level']      = elem['level']
                elif split == 'train':
                    curr_dict['level']      = 'i.i.d.'
                elif split == 'test':
                    curr_dict['level']      = 'zero-shot'

                qonly_data.append(curr_dict)
            
            with open(question_only_data_path, 'w') as f:
                json.dump(qonly_data, f, indent=4)
            
            print(f'Filtered data for {dataset} {split} is {len(filtered_iso_data)} ')
            print(f'Original data for {dataset} {split} is {len(iso_data)}')
                
    for dataset in datasets:
        for split in isomorphism_dict[dataset]:
            print(f'Isomorphism counts for {dataset} {split}:')
            pprint(isomorphism_dict[dataset][split])

            

def tiara_coverage():

    datasets = ['grailqa']# 'webqsp', 'graphquestions', 'cwq']

    splits  = {
                # 'webqsp': ['train', 'test'],
                'grailqa': ['dev', 'train', 'test'],
                # 'cwq': ['train', 'test', 'dev'],
                # 'graphquestions': ['train', 'test']
            }


    for dataset in datasets:

        for split in splits[dataset]:

            relation_coverage_dict                             = ddict(list)

            filtered_input_data                                = json.load(open(f'../output_data/{dataset}.{split}.filtered.json'))

            if split in ['dev', 'test']:
                filtered_schema_data                           = f'/data/shire/projects/GenQA/ISOKBQA/KC/papers/TIARA/dataset/GrailQA/schema_linking_results/dense_retrieval_{dataset}_{split}-filtered.jsonl'
            else:
                filtered_schema_data                           = f'/data/shire/projects/GenQA/ISOKBQA/KC/papers/TIARA/dataset/GrailQA/schema_linking_results/dense_retrieval_{dataset}_{split}.jsonl'

            filtered_input_dict                                = {}

            for elem in filtered_input_data:
                filtered_input_dict[str(elem['qid'])]          = elem

            filtered_schema_dict                               = {}
            filtered_schema_data                               = [json.loads(line) for line in open(filtered_schema_data, 'r')]

            for elem in filtered_schema_data:
                filtered_schema_dict[str(elem['qid'])]         = elem

            print(f'Processing {dataset} {split} with {len(filtered_input_data)} queries and {len(filtered_schema_data)} schema links')

            for qid in filtered_input_dict:

                inp_data                                     = filtered_input_dict[qid]
                schema_data                                  = filtered_schema_dict.get(qid, None)

                if schema_data is None:
                    print(f'No schema data for {qid}, skipping...')
                    continue

                assert str(inp_data['qid']) == str(schema_data['qid']), f'QID mismatch for {inp_data["qid"]}'

                query_graph     = inp_data['graph_query']
                nodes, edges    = query_graph['nodes'], query_graph['edges']
                relations       = set([edge['relation'] for edge in edges])

                if split == 'test':
                    classes         = [node['id'] for node in nodes if node['node_type'] != 'entity' and node['node_type'] != 'literal']
                else:
                    classes         = [node['class'] for node in nodes if node['node_type'] != 'entity' and node['node_type'] != 'literal']

                predicted_classes   = schema_data['classes'][:args.top_k]
                predicted_relations = schema_data['relations'][:args.top_k]

                curr_class_coverage      = 1 - (len(set(classes) - set(predicted_classes)) / len(set(classes)))
                relation_coverage_dict['class coverage'].append(curr_class_coverage)

                curr_relation_coverage   = 1 - (len(set(relations) - set(predicted_relations)) / len(set(relations)))
                relation_coverage_dict['relation coverage'].append(curr_relation_coverage)


            for cat in relation_coverage_dict:
                avg_cov = np.mean(relation_coverage_dict[cat])
                print(f'Average {cat} for {dataset} {split}: {avg_cov:.4f}')




def create_max_subgraphs():

    
    all_tuples_file       = open('../../ontology/fb_tuples')
    triples               = []

    datasets              = ['grailqa']# 'webqsp', 'graphquestions', 'cwq']

    splits                = {'grailqa': ['train', 'dev']}

    rev_flag              = args.rev_flag

    ############# Load all the tuples from the file

    pairwise_rels         = ddict(list)

    for line in all_tuples_file:
        fields = line.strip().split('\t')
        if len(fields) != 3:
            continue
        head, rel, tail = fields

        if rev_flag == 0:
            if rel.startswith('rev_'):
                continue

        triples.append((head, rel, tail))
        if (head, tail) not in pairwise_rels:
            pairwise_rels[(head, tail)] = []
        pairwise_rels[(head, tail)].append(rel)


    G = nx.MultiDiGraph()
    # G = G.to_undirected()

    num_inv_relations       = 0
    for head, rel, tail in triples:
        G.add_edge(head, tail, relation=rel)
        
        if rev_flag ==1:
            if (tail, head) not in pairwise_rels:
                G.add_edge(tail, head, relation= f'rev_{rel}')
                num_inv_relations += 1
            
    print(f'Loaded {len(triples)} tuples with {num_inv_relations} inverse relations')

    # exit()

    subgraph_dict                              = ddict(lambda: ddict(list))
    for dataset in datasets:
        for split in splits[dataset]:

            iso_data_path                      = f'../output_data/{dataset}.{split}.filtered.json'
            iso_data                           = json.load(open(iso_data_path))
            
            cases                              = 0
            print(f'Validating relations for {dataset} {split} with {len(iso_data)} queries')

            for line in tqdm(iso_data):

                query_graph     = line['graph_query']
                nodes, edges    = query_graph['nodes'], query_graph['edges']

                relations       = set([edge['relation'] for edge in edges])
                ent_classes     = [node['class'] for node in nodes if node['node_type'] == 'entity' or node['node_type'] == 'literal']

                node_classes    = set([node['class'] for node in nodes])
                
                # Step 4: Get nodes at k-hop distance from a source

                k_hop_nodes     = []
                k_hop_edges     = []
                k_hop_tuples    = []
                k_hop_relations = []

                for node in ent_classes:
                    try:
                        start_node  = node
                        k           = args.num_hop
                        # shortest_paths = nx.single_source_shortest_path_length(G, start_node)
                        # # import pdb; pdb.set_trace()
                        # k_hop_nodes.extend([node for node, dist in shortest_paths.items() if dist <= k])

                        paths = nx.single_source_shortest_path(G, start_node, cutoff=k)

                        for target_node, path in paths.items():
                            if len(path) <= 1:
                                continue  # Skip the source node itself

                            for i in range(len(path) - 1):
                                u, v = path[i], path[i + 1]
                                # Get all edges u->v, pick the first for now
                                
                                k_hop_nodes.append(u)
                                k_hop_nodes.append(v)
                                k_hop_edges.append((u, v))

                                edge_data = G.get_edge_data(u, v)

                                if edge_data is not None:
                                    poss_relations = len(edge_data)
                                    for idx in range(poss_relations):
                                        rel = edge_data[idx]['relation']
                                        k_hop_tuples.append((u, rel, v))
                                        k_hop_relations.append(rel)
                                    
                                edge_data = G.get_edge_data(v, u)

                                if edge_data is not None:
                                    poss_relations = len(edge_data)
                                    for idx in range(poss_relations):
                                        rel = edge_data[idx]['relation']
                                        k_hop_tuples.append((v, rel, u))
                                        k_hop_relations.append(rel)

                    except Exception as e:
                        cases += 1
                        # import pdb; pdb.set_trace()
                        print(f'Error processing node {node}: {e}')
                        continue

                unique_k_hop_nodes      = set(k_hop_nodes)
                unique_k_hop_edges      = set(k_hop_edges)
                unique_k_hop_tuples     = set(k_hop_tuples)
                unique_k_hop_relations  = set(k_hop_relations)


                # subgraph_dict[dataset][split].append(len(unique_k_hop_nodes))
                node_coverage               = 1 -  len(node_classes - unique_k_hop_nodes) / len(node_classes) 
                rel_coverage                = 1 -  len(relations - unique_k_hop_relations) / len(relations) 

                subgraph_dict[f'{dataset}-{split}']['nodes'].append(len(unique_k_hop_nodes))
                subgraph_dict[f'{dataset}-{split}']['node_coverage'].append(node_coverage)
                subgraph_dict[f'{dataset}-{split}']['edges'].append(len(unique_k_hop_edges))
                subgraph_dict[f'{dataset}-{split}']['tuples'].append(len(unique_k_hop_tuples))
                subgraph_dict[f'{dataset}-{split}']['relations'].append(len(unique_k_hop_relations))
                subgraph_dict[f'{dataset}-{split}']['rel_coverage'].append(rel_coverage)

                # if node_coverage < 1 or rel_coverage < 1:
                #     import pdb; pdb.set_trace()
                

            print(f'Number of cases with errors for {dataset} {split}: {cases}')

    for dataset in subgraph_dict:
        for cat in subgraph_dict[dataset]:
            avg_size = sum(subgraph_dict[dataset][cat]) / len(subgraph_dict[dataset][cat])
            max_size = max(subgraph_dict[dataset][cat])
            min_size = min(subgraph_dict[dataset][cat])

            print(f'Average subgraph {cat} for {dataset} : {avg_size:.4f}')
            print(f'Max subgraph {cat} for {dataset} : {max_size}')
            print(f'Min subgraph {cat} for {dataset} : {min_size}')
            print()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process isomorphism datasets')
    parser.add_argument('--step', required=True, help='Step to perform')
    parser.add_argument('--num_hop', required=False, default=3, type=int, help='Number of hops for k-hop subgraph')
    parser.add_argument('--rev_flag',required=False, default=1, type=int, help='Flag for reverse relations')
    parser.add_argument('--top_k',required=False, default=25, type=int, help='Flag for reverse relations')


    args = parser.parse_args()

    if args.step == 'create':
        create_isomorphism_datasets()
    
    if args.step == 'val_rel':
        validate_relations()
    
    if args.step == 'filter_grailqa':
        filter_grailqa_isomorphisms()
    
    if args.step == 'max_subgraph':
        create_max_subgraphs()
    
    if args.step == 'tiara_cov':
        tiara_coverage()