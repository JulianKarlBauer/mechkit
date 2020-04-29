import itertools

fixed = ["E1", "E2"]
variable = ["G12", "G23", "nu12", "nu21", "nu23"]
variable_combinations = list(itertools.combinations(variable, 3))

combinations = [fixed + list(comb) for comb in variable_combinations]
