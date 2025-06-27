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

s_expr = "(JOIN (R government.government_position_held.office_holder) (AND (JOIN government.government_position_held.time_macro 2015^^http://www.w3.org/2001/XMLSchema#date) (AND (JOIN government.government_position_held.office_position_or_title m.0j5_3sz) (JOIN (R government.governmental_jurisdiction.governing_officials) m.06mkj))))"

s_expr_2="(JOIN (R government.government_position_held.office_holder) (TC (AND (JOIN (R government.governmental_jurisdiction.governing_officials) m.06mkj) (JOIN government.government_position_held.office_position_or_title m.0j5_3sz)) government.government_position_held.from NOW))",

print(obtain_isomorphism_id(s_expr))
print(obtain_isomorphism_id(s_expr_2))


print(replace_s_expr(s_expr))


# print(replace_s_expr(s_expr_2))