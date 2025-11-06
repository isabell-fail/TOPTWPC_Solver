"""
utils.py
========

This module provides utility functions for reading, preprocessing, and postprocessing
instances of the Time-Oriented Team Orienteering Problem with Priorities and Constraints
(TOPTWPC).

It is responsible for:
    1. Parsing raw instance files into structured node data and priorities.
    2. Building a distance matrix that combines travel times and service durations.
    3. Extracting attribute vectors such as deadlines, start times, and profits.
    4. Preparing all inputs required by the heuristic and optimization layers.
    5. Extracting vehicle routes from a solved Gurobi model.

The functions here are used by both the heuristic layer (`heuristics.py`)
and the exact optimization model (`toptwpc.py`) via `main.py`.

Key Features:
-------------
- Instance parsing (`parse_instance`) following the standard TOPTW-PC format:
  * header with global parameters,
  * per-node coordinates, service times, profits, and time windows,
  * final line containing node priorities.
- Construction of a dense distance matrix with:
  * Euclidean distances,
  * travel time + service duration at the target node,
  * special handling of depot–depot arcs.
- Convenience extractors for:
  * deadlines,
  * start times,
  * service times,
  * node profits.
- High-level preprocessing entry point:
  * `process_instance` → returns all data needed for model/heuristics.
- Post-solver route extraction:
  * `extract_solution` → reconstructs vehicle routes from Gurobi x-variables.

Dependencies:
-------------
- External libraries:
  - `numpy` for numeric arrays and distance matrix construction.
- Python standard library:
  - `typing` (if type hints are used elsewhere).

Main Functions:
---------------
- `parse_instance(name)`
    → Reads a TOPTWPC instance file and returns node data, number of vehicles, and priorities.
- `calculate_distance_matrix(nodes, name)`
    → Builds the travel-time + service-time distance matrix.
- `extract_deadlines(nodes)`
    → Extracts closing times (deadlines) for all nodes.
- `extract_start_times(nodes)`
    → Extracts opening times for all nodes.
- `extract_service_times(nodes)`
    → Extracts service durations for all nodes.
- `extract_profits(nodes)`
    → Extracts profits for all non-depot nodes.
- `process_instance(name)`
    → End-to-end preprocessing: instance → matrix, deadlines, profits, priorities, #vehicles.
- `extract_solution(model, num_vehicles)`
    → Rebuilds routes (one per vehicle) from a solved Gurobi model.

Notes:
------
- Node data is read as `(i, x, y, d, S, O, C)` per line, where `i` is the node index.
- The first node acts as both start and end depot: it is duplicated at the end of the `nodes`
  list so that index 0 and index -1 both refer to the depot.
- Profits are extracted only for internal nodes (excluding the depot at the beginning and end),
  consistent with the indexing conventions in `heuristics.py` and `toptwpc.py`.
"""



from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

# Node tuple:
# (index, x, y, service_time, profit, window_open, window_close)
Node = Tuple[int, float, float, float, float, int, int]


def parse_instance(name: str) -> Tuple[List[Node], int, List[int]]:
    """
    Parse a TOPTWPC instance file and return nodes, number of vehicles and priorities.

    Parameters:
    - name: Path to the instance file.

    Returns:
    - nodes: List of nodes as tuples (i, x, y, d, profit, open, close).
    - num_vehicles: Number of vehicles (v from the file header).
    - priorities: List of priorities for each node (from the last line).
    """
    # Read the file
    with open(name, "r", encoding="utf-8") as file:
        lines = file.readlines()

    # Parse the initial information
    # k, v, N, t: only v (number of paths) is relevant as number of vehicles
    _, num_vehicles, _, _ = map(int, lines[0].split())
    # D, Q are not relevant for this implementation
    _ = list(map(float, lines[1].split()))

    nodes: List[Node] = []

    # Parse the remaining lines for node data (excluding the last line with priorities)
    for line in lines[2:-1]:
        parts = line.split()
        if not parts:
            continue

        idx = int(parts[0])          # vertex number
        x_coord = float(parts[1])    # x coordinate
        y_coord = float(parts[2])    # y coordinate
        service_time = float(parts[3])  # service duration
        profit = float(parts[4])        # profit
        window_open = int(parts[-2])    # opening of time window
        window_close = int(parts[-1])   # closing of time window

        nodes.append((idx, x_coord, y_coord, service_time, profit, window_open, window_close))

    # Priorities from last line
    priorities = list(map(int, lines[-1].split()))

    # Add the depot to the end of the list as duplicate of the first node
    nodes = nodes + [nodes[0]]

    return nodes, num_vehicles, priorities


def calculate_distance_matrix(nodes: List[Node], name: str) -> np.ndarray:
    """
    Calculate the distance/travel-time matrix between all nodes.

    Parameters:
    - nodes: List of nodes.
    - name: Name of the instance (used to select rounding precision).

    Returns:
    - distance_matrix: num_nodes x num_nodes matrix of travel costs.
    """
    num_nodes = len(nodes)
    distance_matrix = np.zeros((num_nodes, num_nodes), dtype=float)
    depot_closing_time = 1000  # Arbitrary large cost between depots

    # Standard benchmark convention: "pr" instances use 2 decimal places
    rounding_precision = 2 if "pr" in name else 1

    for i in range(num_nodes):
        xi, yi = nodes[i][1], nodes[i][2]

        for j in range(num_nodes):
            if i == j:
                distance_matrix[i, j] = 0.0
                continue

            xj, yj = nodes[j][1], nodes[j][2]
            distance = ((xi - xj) ** 2 + (yi - yj) ** 2) ** 0.5
            # Travel time + service time of target node
            travel_cost = round(distance, rounding_precision) + nodes[j][3]
            distance_matrix[i, j] = travel_cost

    # Set the distances between the depots to a very high value
    distance_matrix[0, -1] = depot_closing_time
    distance_matrix[-1, 0] = depot_closing_time

    return distance_matrix


def extract_deadlines(nodes: List[Node]) -> List[int]:
    """
    Extract deadline (closing time) for all nodes.

    Parameters:
    - nodes: List of nodes.

    Returns:
    - List of closing times C for each node.
    """
    return [node[6] for node in nodes]


def extract_start_times(nodes: List[Node]) -> List[int]:
    """
    Extract opening times of all nodes.

    Parameters:
    - nodes: List of nodes.

    Returns:
    - List of opening times O for each node.
    """
    return [node[5] for node in nodes]


def extract_service_times(nodes: List[Node]) -> List[float]:
    """
    Extract service times of all nodes.

    Parameters:
    - nodes: List of nodes.

    Returns:
    - List of service durations d for each node.
    """
    return [node[3] for node in nodes]


def extract_profits(nodes: List[Node]) -> List[float]:
    """
    Extract profits for all non-depot nodes.

    Parameters:
    - nodes: List of nodes.

    Returns:
    - List of profits S for nodes 1..n-2 (excluding first and last depot).
    """
    return [node[4] for node in nodes][1:-1]


def process_instance(
    filepath: str,
) -> Tuple[np.ndarray, List[int], List[float], List[int], int, List[int]]:
    """
    Process an instance file into model-ready structures.

    Parameters:
    - filepath: Path to the instance file.

    Returns:
    - distance_matrix: Travel-time matrix.
    - deadlines: List of closing times for each node.
    - profits: List of profits for non-depot nodes.
    - priorities: List of priorities per node.
    - num_vehicles: Number of vehicles.
    - start_times: List of opening times for each node.
    """
    nodes, num_vehicles, priorities = parse_instance(filepath)
    distance_matrix = calculate_distance_matrix(nodes, filepath)
    deadlines = extract_deadlines(nodes)
    profits = extract_profits(nodes)
    start_times = extract_start_times(nodes)

    return distance_matrix, deadlines, profits, priorities, num_vehicles, start_times


def extract_solution(model: Any, num_vehicles: int) -> Dict[int, List[int]]:
    """
    Extract vehicle routes from a solved TOPTWPC model (without explicit vehicle index in x).

    Parameters:
    - model: Gurobi model instance with x[i,j] variables.
    - num_vehicles: Number of vehicles.

    Returns:
    - vehicle_routes: Dict mapping vehicle index -> route (list of node indices).
    """
    edges: List[Tuple[int, int]] = []
    for var in model.getVars():
        if var.VarName.startswith("x") and var.X > 0.5:
            i_str, j_str = var.VarName.replace("x[", "").replace("]", "").split(",")
            i, j = int(i_str), int(j_str)
            edges.append((i, j))

    vehicle_routes: Dict[int, List[int]] = {vehicle: [1] for vehicle in range(num_vehicles)}

    # Greedily connect edges to form routes for each vehicle
    while edges:
        progress = False
        for (i, j) in edges[:]:  # iterate over a copy to avoid modifying while iterating
            for vehicle in range(num_vehicles):
                if vehicle_routes[vehicle][-1] == i:
                    vehicle_routes[vehicle].append(j)
                    edges.remove((i, j))
                    progress = True
                    break
        if not progress:
            # No further edges can be attached to routes; break to avoid infinite loop.
            break
    return vehicle_routes
