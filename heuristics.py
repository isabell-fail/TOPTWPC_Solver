"""
heuristics.py
=============

This module provides a suite of heuristic and metaheuristic algorithms designed to generate
and improve solutions for the Time-Oriented Team Orienteering Problem with Priorities and
Constraints (TOPTWPC).

The algorithms implemented here combine greedy heuristics, local optimization, and metaheuristic
techniques (such as simulated annealing and 2-opt / 3-opt improvements) to efficiently construct
and refine routing plans that maximize profit while respecting time windows, priorities,
and deadlines.

The main entry point is the `full_heuristic()` function, which orchestrates the following phases:
    1. Greedy initialization using a priority- and profit-based heuristic.
    2. Simulated annealing optimization for probabilistic improvement.
    3. Local optimization leveraging the `TOPTWPC` mathematical model for fine-tuning.

Key Features:
--------------
- Priority-based route generation with cumulative profit adjustments.
- Greedy heuristics for initial feasible solutions.
- Simulated annealing with 2-opt and 3-opt neighborhood search.
- Local optimization for subroutes using Gurobi-based TOPTWPC models.
- Utility functions for manipulating routes, edges, and feasibility checks.

Dependencies:
--------------
- Python standard library: `math`, `random`
- Custom module: `TOPTWPC` (imported from `toptwpc`)
- External solver (e.g., Gurobi) through the TOPTWPC interface

Main Functions:
---------------
- `full_heuristic(matrix, num_vehicles, profits, priorities, deadlines)`
    → Generates a near-optimal routing plan using combined heuristics.
- `local_optimization(...)`
    → Refines existing routes using subproblem extraction and model-based optimization.
- `simulated_annealing(...)`
    → Improves routes through probabilistic exploration of neighboring solutions.
- `greedy_heuristic_smart(...)`
    → Builds an initial feasible set of routes based on priorities and profits.
- Additional helper functions for feasibility checking, route updates, and matrix transformations.

Notes:
------
This module assumes that node indexing begins at 1, with node 1 representing the depot.
Profits and priorities are indexed from node 2 onward to align with problem conventions.
"""

# pylint: disable=too-many-lines

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

from toptwpc import TOPTWPC

Matrix = List[List[int]]
Route = List[int]
Routes = List[Route]


def full_heuristic(
        matrix: Matrix,
        num_vehicles: int,
        profits: List[int],
        priorities: List[int],
        deadlines: List[int],
) -> Tuple[List[Tuple[int, int]], List[int]]:
    """
    Optimized heuristic for generating an initial solution.

    Parameters:
    - matrix: 2D list of ints, distance matrix between nodes
    - num_vehicles: int, number of vehicles available
    - profits: list of int, profit values for visiting each evacuation node
    - priorities: list of int, priority values for evacuation nodes
    - deadlines: list of int, deadlines for nodes

    Returns:
    - edges: list of tuples, edges representing the routes of vehicles
    - nodes: list of int, visited nodes in the final routes
    """

    # Step 0: Preprocessing with optimized cumulative sums for alpha
    sum_dict: Dict[int, int] = {}
    for i in range(1, max(priorities) + 1):
        sum_dict[i] = sum(profits[s] for s in range(len(profits)) if priorities[s] == i)
    alpha: List[int] = []
    for _, priority in enumerate(priorities):
        if priority == 1:
            alpha.append(1)
        else:
            cumulative_sum = sum(sum_dict[s] for s in range(1, priority))
            alpha.append(max(cumulative_sum, 1))

    adjusted_profits = [p + a for p, a in zip(profits, alpha)]

    # Step 1: Greedy Heuristic
    routes, greedy_score = greedy_heuristic_smart(
        matrix, num_vehicles, adjusted_profits, priorities, deadlines
    )

    # Step 2: Simulated Annealing
    _, routes = simulated_annealing(
        matrix,
        num_vehicles,
        adjusted_profits,
        priorities,
        deadlines,
        routes,
        greedy_score,
    )

    # Step 3: Local Optimization
    routes = local_optimization(
        matrix,
        num_vehicles,
        adjusted_profits,
        priorities,
        deadlines,
        routes,
    )

    # Step 4: Extract edges, nodes, and calculate the final total score
    edges: List[Tuple[int, int]] = []
    nodes: set[int] = set()
    for route in routes:
        nodes.update(route[1:-1])  # Collect nodes (skip the depots)
        edges.extend((route[i], route[i + 1]) for i in range(len(route) - 1))  # Create edges

    return edges, sorted(nodes)


def greedy_heuristic_smart(
        matrix: Matrix,
        num_vehicles: int,
        profits: List[int],
        priorities: List[int],
        deadlines: List[int],
) -> Tuple[Routes, float]:
    """
    A hybrid heuristic that starts with priority-based assignment and then redistributes nodes
    from overloaded vehicles to underutilized ones.

    Parameters:
    - matrix (2D array): A distance matrix representing travel times between nodes.
    - num_vehicles (int): Number of vehicles/teams available.
    - profits (list): List of scores or profits for visiting each node.
    - priorities (list): List of priorities for each node (higher values indicate higher
      priority).
    - deadlines (list): List of deadlines for each node (maximum time a node can be visited).

    Returns:
    - routes (list of lists): Each list represents the sequence of nodes visited by a vehicle.
    - total_score (float): Total score collected by all vehicles.
    """

    num_nodes = len(matrix)
    routes: Routes = [[1] for _ in range(num_vehicles)]
    total_score: float = 0.0
    evacuation_nodes: List[Tuple[int, int, int, int]] = [
        (i, priorities[i - 2], profits[i - 2], deadlines[i - 1]) for i in range(2, num_nodes)
    ]
    evacuation_nodes.sort(key=lambda x: (-x[1], -x[2]))  # Sort by priority then by profit

    # Initial assignment based on priority and feasibility
    for node_idx, _, profit, _ in evacuation_nodes:
        # Try to assign the node to one of the vehicles
        for vehicle_id in range(1, num_vehicles + 1):
            current_route = routes[vehicle_id - 1]
            longer_route = current_route + [node_idx]

            if (
                    deadline_feasible(longer_route + [num_nodes], matrix, deadlines)
                    and priority_feasible(longer_route + [num_nodes], priorities)
            ):
                routes[vehicle_id - 1].append(node_idx)
                total_score += profit
                break

    for route in routes:
        route.append(num_nodes)

    max_iterations = 100
    iteration_count = 0
    change = True

    while change and iteration_count < max_iterations:
        change = False
        iteration_count += 1

        max_route = max(routes, key=len)
        min_nodes_threshold = int((len(max_route) - 2) / 2)

        # Classify vehicles based on number of nodes in their routes
        underutilized_vehicles = [
            i for i in range(num_vehicles) if len(routes[i]) - 2 < min_nodes_threshold
        ]
        underutilized_vehicles.sort(key=lambda x: len(routes[x]) - 2)
        other_vehicles = [i for i in range(num_vehicles) if i not in underutilized_vehicles]
        other_vehicles.sort(key=lambda x: len(routes[x]) - 2, reverse=True)

        # Redistribution logic
        for underutilized_vehicle in underutilized_vehicles:
            for overloaded_vehicle in other_vehicles:
                overloaded_route = routes[overloaded_vehicle]
                half_point = len(overloaded_route) // 2
                transfer_nodes = overloaded_route[half_point:-1]

                underutilized_route = (
                        routes[underutilized_vehicle][:-1] + transfer_nodes + [num_nodes]
                )
                remaining_route = overloaded_route[:half_point] + [num_nodes]

                if (
                        deadline_feasible(underutilized_route, matrix, deadlines)
                        and deadline_feasible(remaining_route, matrix, deadlines)
                        and priority_feasible(underutilized_route, priorities)
                        and priority_feasible(remaining_route, priorities)
                        and len(underutilized_route) != len(overloaded_route)
                ):
                    routes[underutilized_vehicle] = underutilized_route
                    routes[overloaded_vehicle] = remaining_route
                    change = True
                    break

    return routes, total_score


def simulated_annealing(
        matrix: Matrix,
        num_vehicles: int,
        profits: List[int],
        priorities: List[int],
        deadlines: List[int],
        initial_routes: Routes,
        initial_score: float,
        max_iterations: int = 1000,
        initial_temperature: float = 100.0,
        cooling_rate: float = 0.99,
) -> Tuple[float, Routes]:
    """
    Optimize routes using simulated annealing with a combination of 2-opt and 3-opt moves.

    Parameters:
    - matrix (2D list): Distance matrix.
    - num_vehicles (int): Number of vehicles in the fleet.
    - profits (list): Profit values for each node.
    - priorities (list): Priority levels for each node.
    - deadlines (list): Deadline values for each node.
    - initial_routes (list of lists): Starting routes for each vehicle.
    - initial_score (float): Initial total score.
    - max_iterations (int): Maximum number of iterations (default is 1000).
    - initial_temperature (float): Starting temperature for annealing (default is 100).
    - cooling_rate (float): Cooling rate for temperature (default is 0.99).

    Returns:
    - tuple: Best score achieved and corresponding routes.
    """
    # Initialize current and best solutions
    current_routes, current_score = initial_routes, initial_score
    best_routes, best_score = current_routes, current_score
    temperature = initial_temperature

    for _ in range(max_iterations):
        # Generate a neighboring solution with 2-opt or 3-opt
        if random.random() < 0.5:
            neighbor_routes = two_opt_sa(matrix, priorities, deadlines, current_routes, temperature)
        else:
            neighbor_routes = three_opt_sa(
                matrix, priorities, deadlines, current_routes, temperature
            )

        # Improve neighbor solution with insertion heuristic
        new_routes, new_score = insertion_heuristic(
            neighbor_routes,
            current_score,
            matrix,
            num_vehicles,
            profits,
            priorities,
            deadlines,
        )
        delta_score = new_score - current_score

        # Accept the new solution if it's better, or probabilistically if it's worse
        if delta_score > 0:
            current_routes, current_score = new_routes, new_score
            if new_score > best_score:
                best_routes, best_score = new_routes, new_score
        else:
            # Accept worse solutions based on the current temperature
            acceptance_probability = math.exp(delta_score / temperature)
            if random.random() < acceptance_probability:
                current_routes, current_score = new_routes, new_score

        # Reduce the temperature
        temperature *= cooling_rate

        # Optional early stopping criterion if temperature is sufficiently low
        if temperature < 1e-3:
            break

    return best_score, best_routes


def two_opt_sa(
        matrix: Matrix,
        priorities: List[int],
        deadlines: List[int],
        routes: Routes,
        temperature: float,
) -> Routes:
    """
    Perform a 2-opt local search with simulated annealing to optimize routes.

    Parameters:
    - matrix (2D list): Distance matrix representing travel times between nodes.
    - priorities (list): Priority levels for each node.
    - deadlines (list): Deadline values for each node.
    - routes (list of lists): List of current routes for each vehicle.
    - temperature (float): Current temperature for simulated annealing.

    Returns:
    - list of lists: Improved routes after 2-opt simulated annealing.
    """
    improved_routes = routes.copy()

    for vehicle_id, route in enumerate(routes):
        num_nodes = len(route)

        for i in range(1, num_nodes - 2):
            for j in range(i + 2, num_nodes):
                # Perform 2-opt swap
                new_route = route[:i] + route[i:j][::-1] + route[j:]

                # Check feasibility based on priorities and deadlines
                if priority_feasible(new_route, priorities) and deadline_feasible(
                        new_route, matrix, deadlines
                ):
                    new_time = calculate_route_time(new_route, matrix, deadlines)
                    old_time = calculate_route_time(route, matrix, deadlines)
                    delta_time = new_time - old_time

                    # Accept new route if it's better or with some probability if worse
                    if delta_time < 0:
                        improved_routes[vehicle_id] = new_route
                        return improved_routes  # First improvement

                    acceptance_probability = math.exp(-delta_time / temperature)
                    if random.random() < acceptance_probability:
                        improved_routes[vehicle_id] = new_route
                        return improved_routes  # Probabilistic acceptance

    return improved_routes


def three_opt_sa(
        matrix: Matrix,
        priorities: List[int],
        deadlines: List[int],
        routes: Routes,
        temperature: float,
) -> Routes:
    """
    Perform a 3-opt swap with simulated annealing to optimize routes.

    Parameters:
    - matrix (2D list): Distance matrix representing travel times between nodes.
    - priorities (list): Priority levels for each node.
    - deadlines (list): Deadline values for each node.
    - routes (list of lists): Current routes for each vehicle.
    - temperature (float): Temperature parameter for simulated annealing.

    Returns:
    - list of lists: Improved routes after applying 3-opt with simulated annealing.
    """

    # pylint: disable=invalid-name
    def three_opt_swap(route: Route, i: int, j: int, k: int) -> List[Route]:
        """
        Generate all possible routes from a 3-opt swap by breaking at i, j, k
        and reconnecting segments.
        """
        A, B, C, D = route[:i], route[i:j], route[j:k], route[k:]
        return [
            A + B[::-1] + C[::-1] + D,  # Reverse B and C
            A + B[::-1] + C + D,  # Reverse B
            A + B + C[::-1] + D,  # Reverse C
            A + C[::-1] + B[::-1] + D,  # Swap and reverse B and C
            A + C + B[::-1] + D,  # Swap B and C, reverse B
            A + C + B + D,  # Swap B and C
            A + C[::-1] + B + D,  # Swap B and C, reverse C
        ]

    improved_routes = routes.copy()

    for vehicle_id, route in enumerate(routes):  # pylint: disable=too-many-nested-blocks
        num_nodes = len(route)

        # Test all combinations of three edges (i, j, k) with constraints on distances
        for i in range(1, num_nodes - 4):
            for j in range(i + 2, num_nodes - 2):
                for k in range(j + 2, num_nodes):
                    new_route_options = three_opt_swap(route, i, j, k)

                    for new_route in new_route_options:
                        improved_routes, accepted = _evaluate_3opt_candidate(
                            route=route,
                            new_route=new_route,
                            vehicle_id=vehicle_id,
                            improved_routes=improved_routes,
                            matrix=matrix,
                            deadlines=deadlines,
                            priorities=priorities,
                            temperature=temperature,
                        )
                        if accepted:
                            return improved_routes

    return improved_routes


def insertion_heuristic(
        routes: Routes,
        total_score: float,
        matrix: Matrix,
        num_vehicles: int,
        profits: List[int],
        priorities: List[int],
        deadlines: List[int],
) -> Tuple[Routes, float]:
    """
    Use an insertion heuristic to add unvisited nodes to routes based on priority and feasibility.

    Parameters:
    - routes (list of lists): Current routes for each vehicle.
    - total_score (float): Current total score (profit).
    - matrix (2D list): Distance matrix.
    - num_vehicles (int): Number of vehicles in the fleet.
    - profits (list): List of profit values for each node.
    - priorities (list): Priority level for each node.
    - deadlines (list): Deadline values for each node.

    Returns:
    - tuple: Updated routes and new total score.
    """
    num_nodes = len(matrix)
    all_nodes = set(range(2, num_nodes))  # Exclude depot node (assumed node 1)
    visited_nodes = {node for route in routes for node in route[1:-1]}  # Exclude depots in routes
    unvisited_nodes = sorted(
        all_nodes - visited_nodes,
        key=lambda x: priorities[x - 2],
        reverse=True,
    )

    new_total_score = total_score

    # Try to insert each unvisited node into the best feasible position in any route
    for node in unvisited_nodes:
        best_insertion: Optional[Tuple[int, int]] = None
        best_insertion_profit = total_score
        regret_value: Optional[float] = None
        node_priority = priorities[node - 2]

        for vehicle_id in range(num_vehicles):
            route = routes[vehicle_id]

            # Evaluate possible insertion points for the node within the current route
            for i in range(1, len(route)):
                prev_node, next_node = route[i - 1], route[i]

                # Priority and feasibility check for insertion
                if (prev_node != 1 and priorities[prev_node - 2] < node_priority) or (
                        next_node != num_nodes and priorities[next_node - 2] > node_priority
                ):
                    continue

                temp_route = route[:i] + [node] + route[i:]
                if priority_feasible(temp_route, priorities) and deadline_feasible(
                        temp_route, matrix, deadlines
                ):
                    new_profit = total_score + profits[node - 2]
                    regret_improvement = new_profit - total_score

                    if regret_value is None or regret_value < regret_improvement:
                        best_insertion = (vehicle_id, i)
                        regret_value = regret_improvement
                        best_insertion_profit = new_profit

        # Apply the best feasible insertion for the current node
        if best_insertion:
            vehicle_id, position = best_insertion
            routes[vehicle_id].insert(position, node)
            visited_nodes.add(node)
            new_total_score = best_insertion_profit

    return routes, new_total_score


def local_optimization(
        matrix: Matrix,
        num_vehicles: int,
        profits: List[int],
        priorities: List[int],
        deadlines: List[int],
        routes: Routes,
) -> Routes:
    """
    Perform a local optimization by iterating through priority levels and adjusting routes.

    Parameters:
    - matrix (2D list): Distance matrix representing travel times between nodes.
    - num_vehicles (int): Number of vehicles in the fleet.
    - profits (list): Profits associated with each node.
    - priorities (list): Priority levels for each node.
    - deadlines (list): Deadline values for each node.
    - routes (list of lists): List of current routes, each a sequence of nodes.

    Returns:
    - list of lists: Updated routes after local optimization.
    """
    priority_levels = sorted(set(priorities), reverse=True)

    for priority in priority_levels:
        # Retrieve nodes and routes by priority
        all_priority_routes: Dict[int, Routes] = {
            p: get_nodes_in_routes_by_priority(routes, priorities, p)
            for p in priority_levels
        }
        actual_priority_nodes = get_priority_nodes(priorities, priority)

        # Skip if all actual priority nodes are already in all routes
        if all(node in route for route in routes for node in actual_priority_nodes):
            continue

        priority_routes = get_nodes_in_routes_by_priority(routes, priorities, priority)

        # Adjust routes to include adjacent nodes and depots as necessary
        for i, priority_route in enumerate(priority_routes):
            if priority_route:
                prev_node, next_node = get_adjacent_nodes(
                    routes,
                    priority_route[0],
                    priority_route[-1],
                )
                priority_routes[i] = [prev_node] + priority_route + [next_node]
            else:
                priority_routes = fill_missing_nodes(
                    priority_routes,
                    all_priority_routes,
                    i,
                    priority_levels,
                    priority,
                    routes,
                    priorities,
                )

            priority_routes = add_depot_to_route(priority_routes, routes, i)

        # Extract priority nodes and calculate required matrices and time differences
        priority_nodes = get_priority_nodes_with_routes(
            matrix,
            priorities,
            priority,
            priority_routes,
        )
        arrival_times = compute_arrival_times(matrix, routes)
        second_nodes: List[Optional[int]] = [
            route[1] if route[1] not in actual_priority_nodes else None
            for route in priority_routes
        ]
        all_second_to_last_nodes: List[int] = [route[-2] for route in priority_routes]
        second_to_last_nodes: List[Optional[int]] = [
            route[-2] if route[-2] not in actual_priority_nodes else None
            for route in priority_routes
        ]

        # Generate submatrices and filtered attributes for optimization
        submatrix = create_submatrix(matrix, priority_nodes)
        subprofits = filter_attributes(profits, priority_nodes, matrix)
        subpriorities = filter_attributes(priorities, priority_nodes, matrix)
        subdeadlines = filter_attributes(deadlines, priority_nodes, matrix, offset=1)
        min_time_differences = calculate_min_time_differences(
            all_second_to_last_nodes,
            routes,
            arrival_times,
            deadlines,
        )

        # Prepare model inputs for optimization
        node_mapping: Dict[int, int] = {i + 1: node for i, node in enumerate(priority_nodes)}
        priority_edges = create_priority_edges(priority_routes)

        # Initialize and optimize the model
        toptwpc = TOPTWPC(submatrix, num_vehicles, subprofits, subpriorities, subdeadlines, k=True)
        toptwpc = set_initial_solution(
            toptwpc,
            priority_edges,
            node_mapping,
            routes,
            second_nodes,
            second_to_last_nodes,
            arrival_times,
            min_time_differences,
            all_second_to_last_nodes,
        )

        toptwpc.model.setParam("LogToConsole", 0)
        toptwpc.model.setParam("TimeLimit", 10)
        toptwpc.model.optimize()

        # Extract optimized routes and update original routes if the solution is improved
        optimal_subroutes = extract_subroutes(toptwpc, num_vehicles, node_mapping)
        objective_values = get_objective_values(toptwpc)
        if not all(x == objective_values[0] for x in objective_values):
            routes = update_routes(routes, optimal_subroutes, priority_routes)

    return routes


def set_initial_solution(
        toptwpc: TOPTWPC,
        priority_edges: List[Tuple[int, Tuple[int, int]]],
        node_mapping: Dict[int, int],
        routes: Routes,
        second_nodes: Sequence[Optional[int]],
        second_to_last_nodes: Sequence[Optional[int]],
        arrival_times: Dict[int, Dict[int, int]],
        min_time_differences: Dict[int, Optional[float]],
        all_second_to_last_nodes: Sequence[int],
) -> TOPTWPC:
    """
    Set the initial solution for the optimization model by assigning initial values to decision
    variables.
    """
    for v in toptwpc.model.getVars():
        var_name = v.VarName
        if var_name.startswith("x"):
            _init_x_var(v, var_name, node_mapping, priority_edges)
        elif var_name.startswith("y"):
            _init_y_var(v, var_name, node_mapping, routes, second_nodes, second_to_last_nodes)
        elif var_name.startswith("t"):
            _init_t_var(
                v,
                var_name,
                node_mapping,
                routes,
                second_nodes,
                all_second_to_last_nodes,
                arrival_times,
                min_time_differences,
            )

    return toptwpc


def _init_x_var(
        v: object,
        var_name: str,
        node_mapping: Dict[int, int],
        priority_edges: List[Tuple[int, Tuple[int, int]]],
) -> None:
    """
    Initialize the start value for binary edge-decision variables `x[i,j,k]` in the optimization
    model.
    """
    i, j, k = map(int, var_name[2:-1].split(","))
    if (k, (node_mapping[i], node_mapping[j])) in priority_edges:
        v.Start = 1
    else:
        v.Start = 0


def _init_y_var(
        v: object,
        var_name: str,
        node_mapping: Dict[int, int],
        routes: Routes,
        second_nodes: Sequence[Optional[int]],
        second_to_last_nodes: Sequence[Optional[int]],
) -> None:
    """
    Initialize binary node-decision variables `y[i,k]` that indicate whether node `i` is visited
    by vehicle `k`.
    """
    i, k = map(int, var_name[2:-1].split(","))
    route_nodes = routes[k - 1]
    mapped_node = node_mapping[i]
    if mapped_node in route_nodes:
        if mapped_node in second_nodes or mapped_node in second_to_last_nodes:
            v.lb, v.ub = 1, 1
        v.Start = 1
    else:
        v.Start = 0


def _init_t_var(
        v: object,
        var_name: str,
        node_mapping: Dict[int, int],
        routes: Routes,
        second_nodes: Sequence[Optional[int]],
        all_second_to_last_nodes: Sequence[int],
        arrival_times: Dict[int, Dict[int, int]],
        min_time_differences: Dict[int, Optional[float]],
) -> None:
    """
    Initialize continuous time variables `t[i,k]` representing the arrival time of vehicle `k`
    at node `i` in the optimization model.
    """
    i, k = map(int, var_name[2:-1].split(","))
    route_nodes = routes[k - 1]
    mapped_node = node_mapping[i]
    if i == 1 or mapped_node not in route_nodes:
        return
    if mapped_node in second_nodes:
        if arrival_times[k - 1].get(mapped_node):
            v.lb = arrival_times[k - 1][mapped_node]
    elif mapped_node in all_second_to_last_nodes:
        if arrival_times[k - 1].get(mapped_node):
            v.ub = (arrival_times[k - 1][mapped_node]
                    + (min_time_differences.get(mapped_node, 0) or 0))


def extract_subroutes(
        toptwpc: TOPTWPC,
        num_vehicles: int,
        node_mapping: Dict[int, int],
) -> Routes:
    """
    Extract subroutes from the optimized model's solution.
    """
    subroutes_edges: Dict[int, List[Tuple[int, int]]] = {i: [] for i in range(1, num_vehicles + 1)}
    for (i, j, k), var in toptwpc.X.items():
        if var.X > 0.0001:  # Check if edge is part of the solution
            subroutes_edges[k].append((i, j))

    return [build_route_from_edges(edges, node_mapping) for edges in subroutes_edges.values()]


def get_objective_values(toptwpc: TOPTWPC) -> List[float]:
    """
    Retrieve the objective values for all solutions in the model's solution pool.
    """
    objective_values: List[float] = []
    for i in range(toptwpc.model.SolCount):
        toptwpc.model.setParam("SolutionNumber", i)
        objective_values.append(toptwpc.model.PoolObjVal)
    return objective_values


def update_routes(
        routes: Routes,
        optimal_subroutes: Routes,
        priority_routes: Routes,
) -> Routes:
    """
    Update main routes with the optimal subroutes found.
    """
    for i, subroute in enumerate(optimal_subroutes):
        if len(routes[i]) > 2:
            # Remove start and end nodes from subroute and old priority subroute
            subroute = subroute[1:-1]
            old_subroute = priority_routes[i][1:-1]

            # Replace old subroute in main route with optimized subroute
            start_index = routes[i].index(old_subroute[0])
            end_index = routes[i].index(old_subroute[-1])
            routes[i] = routes[i][:start_index] + subroute + routes[i][end_index + 1:]
        else:
            # Case for routes with only depot nodes (e.g., [1, depot])
            subroute = subroute[1:-1]
            routes[i] = [routes[i][0]] + subroute + [routes[i][-1]]

    return routes


def create_priority_edges(priority_routes: Routes) -> List[Tuple[int, Tuple[int, int]]]:
    """
    Generate edges from priority routes.
    """
    priority_edges: List[Tuple[int, Tuple[int, int]]] = []
    for i, route in enumerate(priority_routes):
        for j in range(len(route) - 1):
            edge = (route[j], route[j + 1])
            priority_edges.append((i + 1, edge))
    return priority_edges


def build_route_from_edges(
        route_edges: List[Tuple[int, int]],
        node_mapping: Dict[int, int],
) -> Route:
    """
    Build a complete route from a list of edges.
    """
    edge_dict = dict(route_edges)
    route: Route = [1]  # Start from node 1 as specified
    current_node = 1

    # Traverse the edges to build the route
    while current_node in edge_dict:
        current_node = edge_dict[current_node]
        route.append(current_node)

    # Use list comprehension to map nodes
    return [node_mapping[node] for node in route]


def compute_arrival_times(matrix: Matrix, routes: Routes) -> Dict[int, Dict[int, int]]:
    """
    Compute arrival times for each node in each route.
    """
    arrival_times: Dict[int, Dict[int, int]] = {}
    for route_idx, route in enumerate(routes):
        current_time = 0
        route_arrival_times: Dict[int, int] = {route[0]: current_time}

        for i in range(1, len(route)):
            current_time += matrix[route[i - 1] - 1][route[i] - 1]
            route_arrival_times[route[i]] = current_time

        arrival_times[route_idx] = route_arrival_times
    return arrival_times


def calculate_route_time(
        route: Route,
        matrix: Matrix,
        deadlines: List[int],
) -> float:
    """
    Calculate the total travel time for a route, returning infinity if deadlines are violated.
    """
    total_time = 0
    for i in range(1, len(route)):
        prev_node, curr_node = route[i - 1], route[i]
        total_time += matrix[prev_node - 1][curr_node - 1]
        if total_time > deadlines[curr_node - 1]:  # Deadline constraint
            return float("inf")
    return float(total_time)


def calculate_total_score(routes: Routes, profits: List[int]) -> int:
    """
    Calculate the total profit score for a set of routes.
    """
    total_score = 0
    for route in routes:
        for node in route[1:-1]:  # Exclude depot (assumed to be first and last in each route)
            total_score += profits[node - 2]  # Use node - 2 to match profit indexing
    return total_score


def priority_feasible(route: Route, priorities: List[int]) -> bool:
    """
    Check if the route respects the priority order constraints between nodes.
    """
    temp_priorities = [math.inf] + priorities + [-math.inf]

    for i in range(2, len(route) - 1):
        current_priority = temp_priorities[route[i] - 1]
        previous_priority = temp_priorities[route[i - 1] - 1]
        next_priority = temp_priorities[route[i + 1] - 1]

        # Check if the current priority level is between the previous and next priority
        if current_priority > previous_priority or current_priority < next_priority:
            return False
    return True


def deadline_feasible(
        route: Route,
        matrix: Matrix,
        deadlines: List[int],
) -> bool:
    """
    Check if a given route respects the deadline constraints for each node.
    """
    total_time = 0
    for i in range(1, len(route)):
        prev_node, curr_node = route[i - 1], route[i]
        total_time += matrix[prev_node - 1][curr_node - 1]  # Accumulate travel time
        if total_time > deadlines[curr_node - 1]:
            return False
    return True


def get_priority_nodes(priorities: List[int], priority: int) -> List[int]:
    """
    Get nodes with a specified priority.
    """
    return [node for node, p in enumerate(priorities, start=2) if p == priority]


def get_nodes_in_routes_by_priority(
        routes: Routes,
        priorities: List[int],
        priority: int,
) -> Routes:
    """
    Get nodes from each route that match a specified priority.
    """
    return [
        [node for node in route[1:-1] if priorities[node - 2] == priority]
        for route in routes
    ]


def get_adjacent_nodes(
        routes: Routes,
        first_node: int,
        last_node: int,
) -> Tuple[Optional[int], Optional[int]]:
    """
    Find the nodes adjacent to the specified nodes within given routes.
    """
    for route in routes:
        if first_node in route and last_node in route:
            idx, idx2 = route.index(first_node), route.index(last_node)
            prev_node = route[idx - 1] if idx > 0 else None
            next_node = route[idx2 + 1] if idx2 < len(route) - 1 else None
            return prev_node, next_node
    return None, None


def fill_missing_nodes(
        priority_routes: Routes,
        all_priority_routes: Dict[int, Routes],
        i: int,
        priority_levels: List[int],
        priority: int,
        routes: Routes,
        priorities: List[int],
) -> Routes:
    """
    Fill missing nodes in priority routes based on adjacent priority levels.
    """
    if not priority_routes[i]:
        # Check lower priority levels in reverse order
        for p in reversed(priority_levels[: priority_levels.index(priority)]):
            if all_priority_routes[p][i]:
                priority_routes[i].insert(0, all_priority_routes[p][i][-1])
                break

        # Check higher priority levels
        for p in priority_levels[priority_levels.index(priority) + 1:]:
            temp_priority_routes = get_nodes_in_routes_by_priority(routes, priorities, p)
            if temp_priority_routes[i]:
                priority_routes[i].append(temp_priority_routes[i][0])
                break

    return priority_routes


def add_depot_to_route(
        priority_routes: Routes,
        routes: Routes,
        i: int,
) -> Routes:
    """
    Ensure that the depot is included at the start and end of each route.
    """
    if 1 not in priority_routes[i]:
        priority_routes[i].insert(0, 1)
    if routes[i][-1] not in priority_routes[i]:
        priority_routes[i].append(routes[i][-1])

    return priority_routes


def get_priority_nodes_with_routes(
        matrix: Matrix,
        priorities: List[int],
        priority: int,
        priority_routes: Routes,
) -> List[int]:
    """
    Retrieve unique priority nodes, including those in priority routes.
    """
    visited = {node for sublist in priority_routes for node in sublist}
    priority_nodes = [node for node in range(2, len(matrix)) if priorities[node - 2] == priority]
    priority_nodes.extend(visited)
    return sorted(set(priority_nodes))


def create_submatrix(matrix: Matrix, priority_nodes: List[int]) -> Matrix:
    """
    Generate a submatrix with only the rows and columns for priority nodes.
    """
    return [
        [matrix[i - 1][j - 1] for j in priority_nodes]
        for i in priority_nodes
    ]


def filter_attributes(
        attributes: List[int],
        priority_nodes: List[int],
        matrix: Matrix,
        offset: int = 2,
) -> List[int]:
    """
    Filter attributes corresponding to specified priority nodes.
    """
    return [
        attributes[node - offset]
        for node in range(offset, len(matrix) + (2 - offset))
        if node in priority_nodes
    ]


def find_min_time_difference(
        node: int,
        routes: Routes,
        arrival_times: Dict[int, Dict[int, int]],
        deadlines: List[int],
) -> Optional[float]:
    """
    Find the minimum positive time difference for a given node in a route.
    """
    # Locate the route and position of the node
    for route_idx, route in enumerate(routes):
        if node in route:
            node_position = route.index(node)
            break
    else:
        return None

    min_difference: float = float("inf")
    for i in range(node_position + 1, len(route)):
        next_node = route[i]
        arrival = arrival_times[route_idx].get(next_node, float("inf"))
        difference = deadlines[next_node - 1] - arrival
        if difference > 0:
            min_difference = min(min_difference, difference)

    return min_difference if min_difference != float("inf") else None


def calculate_min_time_differences(
        second_to_last_nodes: List[int],
        routes: Routes,
        arrival_times: Dict[int, Dict[int, int]],
        deadlines: List[int],
) -> Dict[int, Optional[float]]:
    """
    Calculate minimum time differences for specific nodes.
    """
    min_time_differences: Dict[int, Optional[float]] = {}
    for node in second_to_last_nodes:
        if node == 1:
            min_time_differences[node] = -1.0
        else:
            min_time_differences[node] = find_min_time_difference(
                node, routes, arrival_times, deadlines
            )
    return min_time_differences


def extract_subproblem(
        matrix: Matrix,
        route: Route,
        priority: int,
        priorities: List[int],
        profits: List[int],
        deadlines: List[int],
) -> Tuple[
    Optional[Route],
    Optional[Matrix],
    Optional[List[int]],
    Optional[List[int]],
    Optional[List[int]],
]:
    """
    Extract a subproblem containing nodes with a specific priority.
    """
    indices = [i for i, node in enumerate(route) if priorities[node - 2] == priority]
    if not indices:
        return None, None, None, None, None

    start, end = indices[0], indices[-1] + 1
    subroute = route[start:end]
    submatrix = [[matrix[i - 1][j - 1] for j in subroute] for i in subroute]
    subprofits = [profits[node - 2] for node in subroute]
    subpriorities = [priorities[node - 2] for node in subroute]
    subdeadlines = [deadlines[node - 1] for node in subroute]

    return subroute, submatrix, subprofits, subpriorities, subdeadlines


def integrate_subroute(
        route: Route,
        subroute: Route,
        priority: int,
        priorities: List[int],
) -> Route:
    """
    Integrate a subroute back into the main route based on priority.
    """
    indices = [i for i, node in enumerate(route) if priorities[node - 2] == priority]
    if indices:
        route[indices[0]: indices[-1] + 1] = subroute
    return route


def _evaluate_3opt_candidate(
        route: Route,
        new_route: Route,
        vehicle_id: int,
        improved_routes: Routes,
        matrix: Matrix,
        deadlines: List[int],
        priorities: List[int],
        temperature: float,
) -> Tuple[Routes, bool]:
    """
    Evaluate a 3-opt candidate route and decide whether to accept it based on
    feasibility and simulated annealing acceptance criteria.
    """
    # Check feasibility first
    if not (
            priority_feasible(new_route, priorities)
            and deadline_feasible(new_route, matrix, deadlines)
    ):
        return improved_routes, False

    new_time = calculate_route_time(new_route, matrix, deadlines)
    old_time = calculate_route_time(route, matrix, deadlines)
    delta_time = new_time - old_time

    # Better solution: accept deterministically
    if delta_time < 0:
        improved_routes[vehicle_id] = new_route
        return improved_routes, True

    # Worse solution: accept with some probability
    acceptance_probability = math.exp(-delta_time / temperature)
    if random.random() < acceptance_probability:
        improved_routes[vehicle_id] = new_route
        return improved_routes, True

    return improved_routes, False
