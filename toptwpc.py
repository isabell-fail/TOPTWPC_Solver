"""
toptwpc.py
==========

This module defines the mathematical optimization model for the
Time-Oriented Team Orienteering Problem with Priorities and Constraints (TOPTWPC),
implemented using Gurobi.

The central class, `TOPTWPC`, encapsulates the full mixed-integer programming (MIP) model,
including:
    - binary decision variables for edges and visited nodes,
    - continuous time variables for node arrival times,
    - objective function with priority-adjusted profits,
    - routing, time-window, and precedence constraints.

In addition, this module provides specialized callback and helper functions that:
    - intercept intermediate solutions during branch-and-bound,
    - extract route fragments and assemble them into feasible tours,
    - apply heuristic improvements via the `heuristics` module,
    - inject improved solutions back into the MIP solver.

Key Features:
-------------
- `TOPTWPC` class for building and managing the Gurobi model.
- Support for two formulations:
  - with explicit vehicle index (`k=True`), using x[i,j,k] / y[i,k] / t[i,k],
  - aggregate model without explicit vehicle index (`k=False`), using x[i,j] / y[i] / t[i].
- Priority-aware objective using profit plus cumulative alpha-weights.
- Time-window constraints with big-M linearization for travel times.
- Edge set pruning based on feasibility and priority precedence rules.
- Sophisticated callback (`callback_without_k`) integrating:
  - relaxed solution inspection,
  - fragment construction,
  - local route improvement using `heuristics.simulated_annealing`,
  - solution injection via `set_solution`.

Dependencies:
-------------
- External libraries:
  - `gurobipy` (Gurobi Python API)
- Python standard library:
  - `collections.defaultdict`
- Local modules:
  - `heuristics` for simulated annealing and route improvement logic

Main Components:
----------------
- `class TOPTWPC`:
    → Wraps a Gurobi model instance and builds the full TOPTWPC formulation.
- `callback_without_k(model, where)`
    → Gurobi callback for the aggregate (no-k) formulation, performing:
      solution inspection, heuristic improvement, and solution injection.
- `get_fragments(edges, model)`
    → Constructs path fragments from fractional or integral edge solutions.
- `extract_routes(edges)`
    → Converts a set of used edges into explicit routes.
- `assign_and_build_tours(fragments, num_vehicles, model)`
    → Assigns fragments to vehicles and builds complete candidate tours.
- `validate_fragment(potential_fragment, model)`
    → Checks time-window and priority feasibility of a route fragment.
- `set_solution(score, model, routes)`
    → Writes a new incumbent solution (x, y) based on candidate routes.

Notes:
------
- Node indexing adheres to the convention:
  - node 1: start depot
  - node n: end depot
  - nodes 2..n-1: customer/evacuation nodes
- Profits and priorities for node i are stored with an offset (often i-2),
  consistent with the conventions in `heuristics.py`.
- Several model attributes (e.g., `_X`, `_Y`, `_E`, `_priorities`) are stored as
  "protected" fields on the Gurobi model object for convenient use inside callbacks.
"""


# pylint: disable=protected-access

from __future__ import annotations

from collections import defaultdict
from typing import DefaultDict, Dict, List, Sequence, Set, Tuple

import gurobipy as grb
from gurobipy import GRB

import heuristics

Matrix = List[List[int]]
Route = List[int]
Routes = List[Route]


class TOPTWPC:
    """
    TOPTWPC model class wrapping a Gurobi model instance.
    """

    # pylint: disable=too-many-instance-attributes, too-few-public-methods, invalid-name

    def __init__(
            self,
            matrix: Matrix,
            num_vehicles: int,
            profits: List[int],
            priorities: List[int],
            deadlines: List[int],
            name: str = "TOPTWPC",
            k: bool = False,
            start: int = 0,
    ) -> None:
        """
        Initialize the TOPTWPC model.

        Parameters:
        - matrix: Distance matrix (n x n).
        - num_vehicles: Number of vehicles.
        - profits: Profits for evacuation nodes (indexed from node 2).
        - priorities: Priorities for evacuation nodes (indexed from node 2).
        - deadlines: Deadlines for all nodes (indexed from node 1).
        - name: Name of the Gurobi model.
        - k: Whether to use vehicle-indexed variables x[i,j,k], y[i,k], t[i,k].
        - start: Lower bound for time variables (earliest possible time).
        """
        # Parameters
        self.matrix: Matrix = matrix
        self.vehicles: List[int] = list(range(1, num_vehicles + 1))
        self.num_vehicles: int = num_vehicles
        self.profits: List[int] = profits
        self.priorities: List[int] = priorities
        self.deadlines: List[int] = deadlines
        self.name: str = name
        self.k: bool = k
        self.start: int = start

        self.n: int = len(matrix)
        self.V: List[int] = list(range(1, self.n + 1))
        self.V_prime: List[int] = list(range(2, self.n))
        self.En: List[Tuple[int, int]] = [(i, j) for i in self.V for j in self.V if i != j]

        biggest_travel_time = 0
        for i in self.V_prime:
            biggest_travel_time = max(
                biggest_travel_time,
                matrix[0][i - 1] + matrix[i - 1][self.n - 1],
            )

        self.T_max: int = max(deadlines[0], biggest_travel_time)
        self.M: int = max(10000, biggest_travel_time + 1)

        # Calculate alpha values for the objective function
        sum_dict: Dict[int, int] = {}
        for i in range(1, max(priorities) + 1):
            sum_dict[i] = sum(profits[s] for s in range(len(profits)) if priorities[s] == i)
        self.alpha: List[int] = []
        for _, priority in enumerate(priorities):
            if priority == 1:
                self.alpha.append(1)
            else:
                cumulative_sum = sum(sum_dict[s] for s in range(1, priority))
                self.alpha.append(max(cumulative_sum, 1))

        # Remove unnecessary edges to create E and E'
        r: Set[Tuple[int, int]] = (
                {(i, 1) for i in self.V if i != 1}
                | {(self.n, j) for j in self.V if j != self.n}
                | {
                    (j, i)
                    for i in self.V_prime
                    for j in self.V_prime
                    if priorities[j - 2] < priorities[i - 2]
                }
                | {
                    (i, j)
                    for i in self.V_prime
                    for j in self.V_prime
                    if (
                    self.matrix[0][i - 1]
                    + self.matrix[i - 1][j - 1]
                    + self.matrix[j - 1][self.n - 1]
                    > self.T_max
            )
                }
        )
        self.E: List[Tuple[int, int]] = [edge for edge in self.En if edge not in r]

        # Decision variables (filled in _initialize_variables)
        self.X: Dict = {}
        self.Y: Dict = {}
        self.T: Dict = {}

        # Gurobi model
        self.model: grb.Model
        self._build_model()

        # Attach data to model for callbacks
        self.model._X = self.X
        self.model._Y = self.Y
        self.model._E = self.E
        self.model._V_prime = self.V_prime
        self.model._priorities = self.priorities
        self.model._deadlines = self.deadlines
        self.model._matrix = self.matrix
        self.model._num_vehicles = self.num_vehicles
        self.model._adjusted_profits = [p + a for p, a in zip(profits, self.alpha)]
        self.model._n = self.n
        self.model._current_best = 0.0

    def _initialize_variables(self) -> None:
        """
        Initialize the decision and auxiliary variables.
        """
        if self.k:
            self.X = self.model.addVars(
                ((i, j, k) for (i, j) in self.E for k in self.vehicles),
                vtype=GRB.BINARY,
                name="x",
            )
            self.Y = self.model.addVars(
                ((i, k) for i in self.V for k in self.vehicles),
                vtype=GRB.BINARY,
                name="y",
            )
            self.T = self.model.addVars(
                ((i, k) for i in self.V for k in self.vehicles),
                vtype=GRB.CONTINUOUS,
                name="t",
            )

        else:
            self.X = self.model.addVars(
                ((i, j) for (i, j) in self.E),
                vtype=GRB.BINARY,
                name="x",
            )
            self.Y = self.model.addVars(
                (i for i in self.V_prime),
                vtype=GRB.BINARY,
                name="y",
            )
            self.T = self.model.addVars(
                (i for i in self.V),
                vtype=GRB.CONTINUOUS,
                name="t",
            )

    def _add_constraints(self) -> None:
        """
        Add constraints to the model.
        """
        if self.k:
            # Constraints 1 and 2: each route starts and ends at the depot
            self.model.addConstrs(
                (
                    grb.quicksum(self.X[1, j, k] for j in self.V if (1, j) in self.E) == 1
                    for k in self.vehicles
                ),
                name="start_from_depot",
            )
            self.model.addConstrs(
                (
                    grb.quicksum(self.X[i, self.n, k] for i in self.V if (i, self.n) in self.E)
                    == 1
                    for k in self.vehicles
                ),
                name="end_at_depot",
            )

            # Constraint 3: each node is visited at most once
            self.model.addConstrs(
                (
                    grb.quicksum(self.Y[i, k] for k in self.vehicles) <= 1
                    for i in self.V_prime
                ),
                name="visit_once",
            )

            # Constraints 4 and 5: auxiliary variable Y is 1 if node i is visited
            edges_dict = {
                j: [(i, j) for i in self.V if (i, j) in self.E and j != self.n] for j in self.V
            }
            edges_dict = {k: v for k, v in edges_dict.items() if v}
            self.model.addConstrs(
                (
                    grb.quicksum(self.X[i, j, k] for (i, j) in edges_dict[j]) == self.Y[j, k]
                    for j in edges_dict.keys()
                    for k in self.vehicles
                ),
                name="visited_if_in_edge_used",
            )
            edges_dict = {
                i: [(i, j) for j in self.V if (i, j) in self.E and i != 1] for i in self.V
            }
            edges_dict = {k: v for k, v in edges_dict.items() if v}
            self.model.addConstrs(
                (
                    grb.quicksum(self.X[i, j, k] for (i, j) in edges_dict[i]) == self.Y[i, k]
                    for i in edges_dict.keys()
                    for k in self.vehicles
                ),
                name="visited_if_out_edge_used",
            )

            # Constraint 6: Time window
            self.model.addConstrs(
                (self.T[i, k] >= self.start for i in self.V for k in self.vehicles),
                name="time_window_lower_bound",
            )
            self.model.addConstrs(
                (self.T[i, k] <= self.deadlines[i - 1] for i in self.V for k in self.vehicles),
                name="time_window_upper_bound",
            )

            # Constraint 7: Time consistency
            self.model.addConstrs(
                (
                    self.T[i, k] + self.matrix[i - 1][j - 1] - self.T[j, k]
                    <= self.M * (1 - self.X[i, j, k])
                    for (i, j) in self.E
                    for k in self.vehicles
                ),
                name="income_time",
            )

        else:
            # Constraints 1 and 2: each route starts and ends at the depot
            self.model.addConstr(
                grb.quicksum(self.X[1, j] for j in self.V if (1, j) in self.E)
                == len(self.vehicles),
                name="start_from_depot",
            )
            self.model.addConstr(
                grb.quicksum(self.X[i, self.n] for i in self.V if (i, self.n) in self.E)
                == len(self.vehicles),
                name="end_at_depot",
            )

            # Constraint 3: each node is visited at most once
            self.model.addConstrs(
                (self.Y[i] <= 1 for i in self.V_prime),
                name="visit_once",
            )

            # Constraints 4 and 5: auxiliary variable Y is 1 if node i is visited
            edges_dict = {
                j: [(i, j) for i in self.V if (i, j) in self.E and j != self.n] for j in self.V
            }
            edges_dict = {k: v for k, v in edges_dict.items() if v}
            self.model.addConstrs(
                (
                    grb.quicksum(self.X[i, j] for (i, j) in edges_dict[j]) == self.Y[j]
                    for j in edges_dict.keys()
                ),
                name="visited_if_in_edge_used",
            )
            edges_dict = {
                i: [(i, j) for j in self.V if (i, j) in self.E and i != 1] for i in self.V
            }
            edges_dict = {k: v for k, v in edges_dict.items() if v}
            self.model.addConstrs(
                (
                    grb.quicksum(self.X[i, j] for (i, j) in edges_dict[i]) == self.Y[i]
                    for i in edges_dict.keys()
                ),
                name="visited_if_out_edge_used",
            )

            # Constraint 6: Time window constraints
            self.model.addConstrs(
                (0 <= self.T[i] for i in self.V),
                name="time_window_lower_bound",
            )
            self.model.addConstrs(
                (self.T[i] <= self.deadlines[i - 1] for i in self.V),
                name="time_window_upper_bound",
            )

            # Constraint 7: Time consistency
            self.model.addConstrs(
                (
                    self.T[i] + self.matrix[i - 1][j - 1] - self.T[j]
                    <= self.M * (1 - self.X[i, j])
                    for (i, j) in self.E
                ),
                name="income_time",
            )

    def _set_objective(self) -> None:
        """
        Set the objective function.
        """
        if self.k:
            self.model.setObjective(
                grb.quicksum(
                    (self.alpha[i - 2] + self.profits[i - 2]) * self.Y[i, k]
                    for i in self.V_prime
                    for k in self.vehicles
                ),
                GRB.MAXIMIZE,
            )
        else:
            self.model.setObjective(
                grb.quicksum(
                    (self.alpha[i - 2] + self.profits[i - 2]) * self.Y[i] for i in self.V_prime
                ),
                GRB.MAXIMIZE,
            )

    def _build_model(self) -> None:
        """
        Build the Gurobi model: create variables, objective, and constraints.
        """
        self.model = grb.Model(self.name)
        self._initialize_variables()
        self._set_objective()
        self._add_constraints()
        self.model.update()


def callback_without_k(model: grb.Model, where: int) -> None:
    """
    Callback function for the model without vehicle-indexed variables (k=False).

    Parameters:
    - model: Gurobi model instance with attached TOPTWPC-related attributes.
    - where: Callback location indicator from Gurobi.
    """
    # pylint: disable=broad-exception-caught
    if where == grb.GRB.Callback.MIPSOL:
        try:
            # Get the solution
            x_sol = model.cbGetSolution(model._X)
            y_sol = model.cbGetSolution(model._Y)

            # Get the edges used
            edges_used: List[Tuple[int, int]] = [
                (i, j) for (i, j) in model._E if x_sol[i, j] > 0.5
            ]

            # Get the nodes visited
            nodes_visited: List[int] = [
                i for i in model._V_prime if y_sol[i] > 0.5
            ]

            # Get the routes from the used edges
            routes = extract_routes(edges_used)

            # Get current score
            score = sum(model._adjusted_profits[i - 2] for i in nodes_visited)

            # Apply simulated annealing
            sa_score, routes = heuristics.simulated_annealing(
                model._matrix,
                model._num_vehicles,
                model._adjusted_profits,
                model._priorities,
                model._deadlines,
                routes,
                score,
            )

            # Get new edges and nodes from routes
            if score < sa_score:
                set_solution(sa_score, model, routes)
        except Exception:
            return

    elif where == grb.GRB.Callback.MIPNODE:
        if model.cbGet(grb.GRB.Callback.MIPNODE_STATUS) == grb.GRB.OPTIMAL:
            try:
                # Get best bound and relaxed score
                if model.cbGet(grb.GRB.Callback.MIPNODE_NODCNT) < 3:
                    return

                y_rel: Dict[int, float] = {
                    i: model.cbGetNodeRel(model._Y[i]) for i in model._V_prime
                }
                relaxed_score = sum(
                    model._adjusted_profits[i - 2] * y for i, y in y_rel.items() if y > 0
                )
                best_bound = model.cbGet(grb.GRB.Callback.MIPNODE_OBJBND)

                # Focus on the best bound
                if relaxed_score != round(best_bound):
                    return

                # Get the edges used
                x_rel = model.cbGetNodeRel(model._X)
                edges: List[Tuple[int, int, float]] = [
                    (i, j, x_rel[i, j]) for (i, j) in model._E if x_rel[i, j] > 1e-5
                ]

                # Focus on solutions with more than 2n edges
                if model._n * 2 < len(edges):
                    return

                # Focus on solutions with less than 15% of edges with value 1
                if sum(1 for (_, _, val) in edges if val == 1) / len(edges) > 0.15:
                    return

                # Sort edges by decision variable value
                edges.sort(key=lambda x: x[2], reverse=True)

                # Find fragments
                fragments = get_fragments(edges, model)

                # Build tours
                routes = assign_and_build_tours(
                    tuple(tuple(f) for f in fragments),
                    model._num_vehicles,
                    model,
                )

                # Get new score
                nodes_visited = [
                    i
                    for route in routes
                    for i in route
                    if i not in (1, model._n)
                ]
                score = sum(model._adjusted_profits[i - 2] for i in nodes_visited)

                if model._current_best < score:
                    set_solution(score, model, routes)
            except Exception:
                return


# pylint: disable=too-many-branches
def get_fragments(
        edges: Sequence[Tuple[int, int, float]],
        model: grb.Model,
) -> List[List[int]]:
    """
    Build path fragments from fractional edges.

    Parameters:
    - edges: List of (i, j, value) triples from the relaxed solution.
    - model: Gurobi model with attached problem data.

    Returns:
    - list of fragments, each fragment a list of node indices.
    """
    node_to_fragments: DefaultDict[int, Set[Tuple[int, ...]]] = defaultdict(set)
    fragments: List[List[int]] = []
    fragments_to_remove: List[List[int]] = []

    for i, j, _value in edges:
        frag_i = [list(frag) for frag in node_to_fragments[i]]
        frag_j = [list(frag) for frag in node_to_fragments[j]]
        common_fragments = frag_i + frag_j if i != 1 and j != model._n else None

        if common_fragments:
            for f1 in frag_i:
                for f2 in frag_j:
                    if f1 == f2:
                        continue  # Skip identical fragments

                    res = False
                    if f1[-1] == i and f2[0] == j:
                        res = add_fragment(f1 + f2, model, fragments, node_to_fragments)
                    elif f2[-1] == i and f1[0] == j:
                        res = add_fragment(f2 + f1, model, fragments, node_to_fragments)

                    if res:
                        fragments_to_remove.append(f1)
                        fragments_to_remove.append(f2)
        else:
            for f in frag_i:
                if f[-1] == i:
                    res = add_fragment(f + [j], model, fragments, node_to_fragments)

                    if res and f not in fragments_to_remove:
                        fragments_to_remove.append(f)

            for f in frag_j:
                if f[0] == j:
                    res = add_fragment([i] + f, model, fragments, node_to_fragments)

                    if res and f not in fragments_to_remove:
                        fragments_to_remove.append(f)

        add_fragment([i, j], model, fragments, node_to_fragments)

    fragments = [frag for frag in fragments if frag not in fragments_to_remove]

    return fragments


def extract_routes(edges: Sequence[Tuple[int, int]]) -> Routes:
    """
    Extract the routes from the set of edges.

    Parameters:
    - edges: List of (i, j) edges.

    Returns:
    - list of routes, each route a list of node indices.
    """
    # Get all edges starting from the depot
    starts = [edge for edge in edges if edge[0] == 1]

    # Get all routes
    routes: Routes = []
    for starting in starts:
        route: Route = [starting[0], starting[1]]
        # Loop until route[-1] is not a beginning of an edge
        while True:
            next_edge = [edge for edge in edges if edge[0] == route[-1]]
            if next_edge:
                route.append(next_edge[0][1])
            else:
                break
        routes.append(route)

    return routes


def add_fragment(
        frag: List[int],
        model: grb.Model,
        fragments: List[List[int]],
        node_to_fragments: DefaultDict[int, Set[Tuple[int, ...]]],
) -> bool:
    """
    Add a fragment to the list and update node mapping if it is valid.

    Returns:
    - True if the fragment was added, False otherwise.
    """
    if validate_fragment(tuple(frag), model):
        fragments.append(frag)
        for node in frag:
            node_to_fragments[node].add(tuple(frag))
        return True
    return False


def validate_fragment(
        potential_fragment: Sequence[int],
        model: grb.Model,
) -> bool:
    """
    Validate a potential fragment.

    Parameters:
    - potential_fragment: Sequence of nodes.
    - model: Gurobi model with attached TOPTWPC data.

    Returns:
    - True if the fragment is valid, False otherwise.
    """
    n: int = model._n
    priorities: List[int] = model._priorities
    deadlines: List[int] = model._deadlines
    matrix: Matrix = model._matrix

    # Check for duplicates using a set
    if len(set(potential_fragment)) != len(potential_fragment):
        return False

    # Start and end check
    if (potential_fragment[0] != 1 and 1 in potential_fragment) or (
            potential_fragment[-1] != n and n in potential_fragment
    ):
        return False

    # Time Window check
    time = matrix[0][potential_fragment[0] - 1] if potential_fragment[0] != 1 else 0
    for k, l in zip(potential_fragment, potential_fragment[1:]):
        time += matrix[k - 1][l - 1]
        if time > deadlines[l - 1]:  # Early exit if deadline is exceeded
            return False

    # Add final leg back to depot
    if potential_fragment[-1] != n:
        time += matrix[potential_fragment[-1] - 1][n - 1]
    if time > deadlines[n - 1]:
        return False

    # Priority check (early exit)
    for k, l in zip(potential_fragment, potential_fragment[1:]):
        if k == 1 or l == n:
            continue
        if priorities[k - 2] < priorities[l - 2]:
            return False  # Early exit if priority fails

    return True


# pylint: disable=too-many-branches
def assign_and_build_tours(
        fragments: Sequence[Sequence[int]],
        num_vehicles: int,
        model: grb.Model,
) -> Routes:
    """
    Assign fragments to vehicles and build complete tours.

    Parameters:
    - fragments: Iterable of fragments (each a sequence of nodes).
    - num_vehicles: Number of vehicles.
    - model: Gurobi model with attached TOPTWPC data.

    Returns:
    - list of routes, one per vehicle.
    """
    fragments_list: List[List[int]] = [list(f) for f in fragments]
    fragment_profits: List[Tuple[List[int], float]] = [
        (
            fragment,
            sum(
                model._adjusted_profits[node - 2]
                for node in fragment
                if node not in (1, model._n)
            ),
        )
        for fragment in fragments_list
    ]
    fragments_list = [
        fragment for fragment, _ in sorted(fragment_profits, key=lambda x: x[1], reverse=True)
    ]

    for fragment in fragments_list.copy():
        if validate_fragment(tuple([1] + fragment + [model._n]), model):
            fragments_list.append([1] + fragment)

    potential_tours: List[List[int]] = [
        fragment
        for fragment in fragments_list
        if fragment[-1] == model._n and fragment[0] == 1
    ]
    fragments_list = [fragment for fragment in fragments_list if fragment not in potential_tours]

    start_fragments: List[List[int]] = sorted(
        (frag for frag in fragments_list if frag[0] == 1 and frag[-1] != model._n),
        key=lambda f: model._priorities[f[-1] - 2],
        reverse=True,
    )
    end_fragments: List[List[int]] = [
                                         fragment for fragment in fragments_list if
                                         fragment[-1] == model._n
                                     ] + [[model._n]]
    other_fragments: List[List[int]] = [
        frag for frag in fragments_list if frag not in start_fragments + end_fragments
    ]

    for start_frag in start_fragments:
        for other_fragment in other_fragments:
            current_tour = start_frag + other_fragment
            if validate_fragment(tuple(current_tour), model):
                potential_tours.append(current_tour)

    potential_full_tours: List[List[int]] = []
    for tour in potential_tours:
        for end_frag in end_fragments:
            candidate = tour + end_frag
            if validate_fragment(tuple(candidate), model):
                potential_full_tours.append(candidate)

    vehicle_tours: Routes = []
    while len(vehicle_tours) < num_vehicles:
        potential_full_tours_with_scores: List[Tuple[List[int], float]] = [
            (
                tour,
                sum(
                    model._adjusted_profits[node - 2]
                    for node in tour
                    if node not in (1, model._n)
                ),
            )
            for tour in potential_full_tours
        ]

        potential_full_tours_with_scores.sort(key=lambda x: x[1], reverse=True)
        best_tour = potential_full_tours_with_scores[0][0]
        vehicle_tours.append(best_tour)
        potential_full_tours.remove(best_tour)

        # Remove all nodes in the best tour from the other tours
        for tour in potential_full_tours:
            for node in best_tour:
                if node in tour and node not in (1, model._n):
                    tour.remove(node)
                    continue

        if not potential_full_tours:
            vehicle_tours.append([1, model._n])
            break

    return vehicle_tours


def set_solution(score: float, model: grb.Model, routes: Routes) -> None:
    """
    Set the solution (routes) in the model as a new incumbent.

    Parameters:
    - score: Score of the solution.
    - model: Gurobi model instance.
    - routes: List of routes (each a list of nodes).
    """
    model._current_best = score
    new_edges: Set[Tuple[int, int]] = set()
    new_nodes: Set[int] = set()
    for route in routes:
        for i in range(len(route) - 1):
            new_edges.add((route[i], route[i + 1]))
            new_nodes.add(route[i])

    # Set new solution
    for v in model.getVars():
        if v.VarName.startswith("x"):
            i_str, j_str = v.VarName.replace("x[", "").replace("]", "").split(",")
            i, j = int(i_str), int(j_str)
            if (i, j) in new_edges:
                model.cbSetSolution(v, 1.0)
            else:
                model.cbSetSolution(v, 0.0)
        elif v.VarName.startswith("y"):
            i = int(v.VarName.replace("y[", "").replace("]", ""))
            if i in new_nodes:
                model.cbSetSolution(v, 1.0)
            else:
                model.cbSetSolution(v, 0.0)
