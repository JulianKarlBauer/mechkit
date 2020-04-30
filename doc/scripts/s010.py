import itertools
from pprint import pprint

items = ["E1", "E2", "G12", "G23", "nu12", "nu21", "nu23"]

combinations = list(itertools.combinations(items, 5))

# Directly required
contain_E1_E2_G12 = [
    comb for comb in combinations if ("E1" in comb and "E2" in comb and "G12" in comb)
]

# Two possibilities to imply G23
contain_E1_E2_G12_imply_G23 = [
    comb for comb in contain_E1_E2_G12 if ("G23" in comb or "nu23" in comb)
]

# Two possibilities to imply nu12
contain_E1_E2_G12_imply_G23_nu12 = [
    comb for comb in contain_E1_E2_G12_imply_G23 if ("nu12" in comb or "nu21" in comb)
]

possible_input = contain_E1_E2_G12_imply_G23_nu12

print('Possible input combinations are')
pprint(possible_input)