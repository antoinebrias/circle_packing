import numpy as np
import matplotlib.pyplot as plt
from pyomo.environ import *
from math import pi

# Dimensions of the area
AREA_WIDTH = 250
AREA_HEIGHT = 100
NUM_CIRCLES = 15
NUM_INITIALIZATIONS = 100  # Number of initializations to test

# Function to create the model
def create_model():
    model = ConcreteModel()
    model.I = RangeSet(1, NUM_CIRCLES)

    model.x = Var(model.I, bounds=(0, AREA_WIDTH), initialize=lambda model, i: np.random.uniform(10, AREA_WIDTH - 10))
    model.y = Var(model.I, bounds=(0, AREA_HEIGHT), initialize=lambda model, i: np.random.uniform(10, AREA_HEIGHT - 10))
    model.r = Var(model.I, bounds=(5, min(AREA_WIDTH, AREA_HEIGHT) / 2), 
                   initialize=lambda model, i: np.random.uniform(5, min(AREA_WIDTH, AREA_HEIGHT) / 2.1))

    # Objective: Maximize the total covered area
    model.covered_area = Objective(expr=sum(pi * model.r[i]**2 for i in model.I), sense=maximize)

    # Constraints to avoid overlap
    def no_overlap_rule(model, i, j):
        if i < j:
            return (model.x[i] - model.x[j])**2 + (model.y[i] - model.y[j])**2 >= (model.r[i] + model.r[j])**2
        return Constraint.Skip

    model.no_overlap = Constraint(model.I, model.I, rule=no_overlap_rule)

    # Constraints to ensure circles stay completely within the area
    def bounds_x_lower_rule(model, i):
        return model.x[i] - model.r[i] >= 0

    def bounds_x_upper_rule(model, i):
        return model.x[i] + model.r[i] <= AREA_WIDTH

    def bounds_y_lower_rule(model, i):
        return model.y[i] - model.r[i] >= 0

    def bounds_y_upper_rule(model, i):
        return model.y[i] + model.r[i] <= AREA_HEIGHT

    model.bounds_x_lower = Constraint(model.I, rule=bounds_x_lower_rule)
    model.bounds_x_upper = Constraint(model.I, rule=bounds_x_upper_rule)
    model.bounds_y_lower = Constraint(model.I, rule=bounds_y_lower_rule)
    model.bounds_y_upper = Constraint(model.I, rule=bounds_y_upper_rule)

    return model

# Function to solve the model
def solve_model(model, solver_name='ipopt'):
    solver = SolverFactory(solver_name)
    results = solver.solve(model, tee=False)
    
    return results

# Test multiple initializations
best_fitness = 0
best_model = None

for _ in range(NUM_INITIALIZATIONS):
    model = create_model()
    results = solve_model(model)

    # Check if the solver found an optimal solution
    if results.solver.termination_condition == TerminationCondition.optimal:
        current_fitness = model.covered_area()
        print(f"Optimal solution found with a covered area of: {current_fitness}")
        
        if current_fitness > best_fitness:
            best_fitness = current_fitness
            best_model = model
    else:
        print("No optimal solution found. Check the model or constraints.")

# Calculate the coverage percentage
if best_model is not None:
    total_area = AREA_WIDTH * AREA_HEIGHT
    coverage_percentage = (best_fitness / total_area) * 100
    print(f"Coverage Percentage: {coverage_percentage:.2f}%")

    circle_positions = [(best_model.x[i].value, best_model.y[i].value, best_model.r[i].value) for i in best_model.I]

    # Plot the best results
    fig, ax = plt.subplots()
    ax.set_xlim(0, AREA_WIDTH)
    ax.set_ylim(0, AREA_HEIGHT)
    ax.set_title('Best Circles Placed in the Area')

    for x, y, r in circle_positions:
        circle = plt.Circle((x, y), r, color='blue', alpha=0.5)
        ax.add_artist(circle)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
