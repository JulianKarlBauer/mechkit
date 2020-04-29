import itertools

fixed = ["G12"]
variable = ['E1', 'E2', "G23", "nu12", "nu21", "nu23"]
variable_combinations = list(itertools.combinations(variable, 4))
combinations = [fixed + list(comb) for comb in variable_combinations]

# Notes: G12 is required!


