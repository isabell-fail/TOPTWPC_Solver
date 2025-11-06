"""
main.py
=======

This module provides the command-line interface (CLI) entry point for solving instances of the
Time-Oriented Team Orienteering Problem with Priorities and Constraints (TOPTWPC).

It orchestrates the full workflow:
    1. Parsing command-line arguments and optional instance help.
    2. Reading and preprocessing instance data via the `utils` module.
    3. Generating an initial heuristic solution using `full_heuristic` from `heuristics`.
    4. Building and solving the corresponding `TOPTWPC` optimization model.
    5. Reporting the best solution and its associated routes.

The module integrates fast constructive heuristics with an exact MIP model (via Gurobi), using the
heuristic solution as a warm start to speed up convergence and improve solution quality.

Key Features:
-------------
- CLI interface with configurable:
  - input instance file
  - number of vehicles
  - solver time limit
  - optional log file name
- Detailed help text describing the instance file format.
- Automatic preprocessing of instances into matrix and attribute structures.
- Use of a heuristic warm start (`full_heuristic`) to initialize the Gurobi model.
- Extraction and printing of vehicle routes from the final solution.

Dependencies:
-------------
- Python standard library: `argparse`, `os`
- Local modules:
  - `heuristics` for heuristic solution construction
  - `toptwpc` for the `TOPTWPC` optimization model and callbacks
  - `utils` for instance parsing and solution extraction
- External solver: Gurobi (accessed through `toptwpc.TOPTWPC`)

Main Functions:
---------------
- `main()`
    → Parses CLI arguments, runs the full TOPTWPC pipeline, and prints solution information.

Usage:
------
This module is intended to be executed as a script, for example:

    python main.py path/to/instance.txt 3 --time_limit 3600 --log_name run1.log

Notes:
------
- The instance format and modeling conventions are aligned with the utilities in `utils.py`.
- Node indexing follows the same convention as in `heuristics` and `toptwpc`:
  node 1 acts as both start and end depot.
"""

import argparse
import os
import sys
from collections.abc import Iterable, Collection, Mapping, Sequence

from heuristics import full_heuristic
from toptwpc import TOPTWPC, callback_without_k
from utils import process_instance, extract_solution


def build_parser() -> argparse.ArgumentParser:
    """
    Create and configure the argument parser for the TOPTWPC command-line interface.

    Returns:
    - argparse.ArgumentParser: Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Löst ein TOPTWPC-Problem basierend auf einer Eingabedatei und einer "
            "Fahrzeuganzahl."
        )
    )
    parser.add_argument("filepath", type=str, nargs="?", help="Pfad zur Eingabedatei.")
    parser.add_argument(
        "num_vehicles",
        type=int,
        nargs="?",
        help="Anzahl der Fahrzeuge.",
    )
    parser.add_argument(
        "--log_name",
        type=str,
        default=None,
        help="Name der Logdatei (optional).",
    )
    parser.add_argument(
        "--time_limit",
        type=int,
        default=10800,
        help=(
            "Zeitlimit für die Optimierung (in Sekunden, optional). "
            "Standard: 10800 Sekunden."
        ),
    )
    parser.add_argument(
        "--instance-help",
        action="store_true",
        help="Erklärung zur Struktur der Instanzdatei anzeigen.",
    )
    return parser


def print_instance_help() -> None:
    """
    Print a detailed explanation of the instance file format.
    """
    print(
        """
************************
* TOPTW-PC test instances *
************************

The first line contains the following data:

\tk v N t

Where
\tk = not relevant
\tv = with this number of paths, all vertices can be visited
\tN = number of vertices
\tt = not relevant

The next line contains the following data:

\tD Q

Where
\tD = not relevant (in many files this number is missing)
\tQ = not relevant

The remaining lines contain the data of each point. 
For each point, the line contains the following data:

\ti x y d S f a list O C

Where
\ti = vertex number
\tx = x coordinate
\ty = y coordinate
\td = service duration or visiting time    
\tS = profit of the location
\tf = not relevant
\ta = not relevant
\tlist = not relevant (length of the list depends on a)
\tO = opening of time window (earliest time for start of service)
\tC = closing of time window (latest time for start of service)

The last line contains the priorities of the points.

* REMARKS *
\t- The first point (index 0) is the starting AND ending point.
\t- The time budget per path (Tmax) equals the closing time of the starting point.
        """
    )


def print_routes(title: str,
                 routes: Mapping[int, Sequence[int]]) -> None:
    """
    Print the routes for all vehicles with a header.

    Parameters:
    - title (str): Header text to print before the routes.
    - routes (dict): Mapping vehicle_id -> route (list of nodes).
    """
    print(title)
    for vehicle, route in routes.items():
        if route:  # Nur Fahrzeuge mit einer Route ausgeben
            print(f"Fahrzeug {vehicle}: {route}")


def warm_start_model(toptwpc: TOPTWPC,
                     edges: Iterable[tuple[int, int]],
                     visited: Collection[int], ) -> None:
    """
    Set the initial solution for the TOPTWPC model based on heuristic edges and visited nodes.

    Parameters:
    - toptwpc (TOPTWPC): Model instance.
    - edges (list of tuples): List of (i, j) edges in the heuristic solution.
    - visited (list or set): Nodes visited in the heuristic solution.
    """
    edge_set = set(edges)
    visited_set = set(visited)

    for var in toptwpc.model.getVars():
        name = var.VarName
        if name.startswith("x"):
            i, j = name[2:-1].split(",")
            i, j = int(i), int(j)
            var.Start = 1 if (i, j) in edge_set else 0
        elif name.startswith("y"):
            i = int(name[2:-1])
            var.Start = 1 if i in visited_set else 0


def configure_model(toptwpc: TOPTWPC, log_name: str, time_limit: int) -> None:
    """
    Configure Gurobi parameters for the TOPTWPC model.

    Parameters:
    - toptwpc (TOPTWPC): Model instance.
    - log_name (str): File name for the log file.
    - time_limit (int): Time limit in seconds.
    """
    toptwpc.model.setParam("LogToConsole", 0)
    toptwpc.model.setParam("TimeLimit", time_limit)
    toptwpc.model.setParam("LogFile", log_name)
    toptwpc.model.setParam("Heuristics", 0.5)
    toptwpc.model.setParam("SimplexPricing", 2)


def run_solver(filepath: str, num_vehicles: int, log_name: str | None, time_limit: int) -> None:
    """
    Run the heuristic warm start and exact TOPTWPC optimization for a given instance.

    Parameters:
    - filepath (str): Path to the instance file.
    - num_vehicles (int): Number of vehicles.
    - log_name (str or None): Optional log file name. If None, a default is constructed.
    - time_limit (int): Time limit in seconds.
    """
    if not os.path.exists(filepath):
        print(f"Die Datei {filepath} wurde nicht gefunden.")
        sys.exit(1)

    # Verarbeite die Eingabedatei
    matrix, deadlines, profits, priorities, _, _ = process_instance(filepath)

    # Hole die Kanten und besuchten Knoten mit der Heuristik
    edges, visited = full_heuristic(matrix, num_vehicles, profits, priorities, deadlines)

    # Erstelle das TOPTWPC-Modell
    toptwpc = TOPTWPC(matrix, num_vehicles, profits, priorities, deadlines, name=filepath)

    # Setze die Kanten und besuchten Knoten als Startlösung
    warm_start_model(toptwpc, edges, visited)

    # Setze die Parameter
    if not log_name:
        log_name = f"{os.path.basename(filepath)}_{num_vehicles}.txt"

    configure_model(toptwpc, log_name, time_limit)

    # Optimieren
    toptwpc.model.optimize(callback_without_k)

    status = toptwpc.model.status

    if status == 2:  # Optimale Lösung gefunden
        print("Optimierung erfolgreich abgeschlossen.")
        print(f"Bester gefundener Score: {toptwpc.model.ObjVal}")
        print("Die gefundene Lösung ist optimal.")

        routes = extract_solution(toptwpc.model, num_vehicles)
        print_routes("Optimale Lösung (Routen):", routes)

    elif status in (9, 13):  # z.B. Zeitlimit erreicht
        print(f"Zeitlimit von {time_limit} Sekunden erreicht!")
        print(f"Bester gefundener Score: {toptwpc.model.ObjVal}")
        gap = toptwpc.model.MIPGap
        print(f"MIP Gap: {gap * 100:.2f}%")

        routes = extract_solution(toptwpc.model, num_vehicles)
        print_routes("Beste gefundene Lösung (Routen):", routes)

    elif status == 3:  # Unlösbar
        print("Das Problem ist unlösbar.")
    else:
        print(f"Optimierung beendet mit Status {status}.")


def main() -> None:
    """
    Entry point for the TOPTWPC command-line interface.

    Parses arguments, optionally prints instance help, and runs the solver.
    """
    parser = build_parser()
    args = parser.parse_args()

    if args.help_instance:
        print_instance_help()
        return

    # Überprüfen, ob die Pflichtargumente fehlen
    if not args.filepath or not args.num_vehicles:
        parser.error(
            "Die Argumente 'filepath' und 'num_vehicles' sind erforderlich, außer "
            "'--instance-help' wird verwendet."
        )

    run_solver(args.filepath, args.num_vehicles, args.log_name, args.time_limit)


if __name__ == "__main__":
    main()
