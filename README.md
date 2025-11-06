# 🧩 TOPTWPC Solver

## Time-Oriented Team Orienteering Problem with Priorities and Constraints

This project provides a **hybrid optimization framework** for solving the **Time-Oriented Team Orienteering Problem with Priorities and Constraints (TOPTWPC)** — a routing problem that combines **profit maximization**, **time windows**, **deadlines**, and **priority-based constraints**.

It integrates **heuristics**, **metaheuristics**, and **exact optimization** via Gurobi to efficiently construct and refine feasible routing solutions.

---

## 🚀 Features

- **Greedy constructive heuristics** for initial solutions  
- **Simulated annealing** and **2-opt / 3-opt** local optimization  
- **Gurobi MIP model** with warm-start from heuristic results  
- **Priority- and time-window-aware** objective function  
- **Callback-based improvement** during optimization

---

## 📂 Project Structure

```text
TOPTWPC/
├── main.py          # CLI entry point for running the solver
├── heuristics.py    # Greedy, SA, and local optimization methods
├── toptwpc.py       # Gurobi MIP model for the TOPTWPC
├── utils.py         # Instance parsing, preprocessing, and solution extraction
└── instances/       # (Optional) Example instance files
```
---

## ⚙️ Installation

**Requirements**
- Python 3.9+
- [Gurobi Optimizer](https://www.gurobi.com/)
- Python dependencies:
  ```bash
  pip install numpy gurobipy

---

## 🧠 Usage

Run the solver from the command line:

```bash
python main.py path/to/instance.txt 3 --time_limit 3600
```
| Argument          | Description                                   |
| ----------------- | --------------------------------------------- |
| `filepath`        | Path to the instance file                     |
| `num_vehicles`    | Number of vehicles                            |
| `--time_limit`    | Solver time limit in seconds (default: 10800) |
| `--log_name`      | Optional log file name                        |
| `--help_instance` | Show detailed instance format help            |

---

## 🧾 Instance References

The included or compatible test instances are based on the **Team Orienteering Problem with Time Windows (TOPTW)** datasets, derived from well-established benchmark sets in the literature. These instances originate from adaptations of the **Solomon (1987)** vehicle routing problems with time windows and the **Cordeau et al. (1997)** multi-depot vehicle routing problems.

### 📚 References

- Righini G., Salani M. *Dynamic programming for the orienteering problem with time windows.*  
  Technical Report 91, Dipartimento di Tecnologie dell’Informazione, Università degli Studi di Milano, Crema, Italy (2006).

- Righini G., Salani M. *New dynamic programming algorithms for the Resource Constrained Elementary Shortest Path.*  
  **Networks**, 51(3), 155–170 (2008).

- Montemanni R., Gambardella L. *Ant Colony System for Team Orienteering Problems with Time Windows.*  
  **Foundations of Computing and Decision Sciences**, 34, 287–306 (2009).

- Vansteenwegen P., Souffriau W., Vanden Berghe G., Van Oudheusden D.  
  *Iterated Local Search for the Team Orienteering Problem with Time Windows.*  
  **Computers & Operations Research**, 36(12), 3281–3290 (2009).  
  [doi:10.1016/j.cor.2009.03.008](https://doi.org/10.1016/j.cor.2009.03.008)

- Souffriau W., Vansteenwegen P., Vanden Berghe G., Van Oudheusden D.  
  *The Multi-Constraint Team Orienteering Problem with Multiple Time Windows.*  
  **Transportation Science**, 47, 53–63 (2013).

- Gunawan A., Lau H.C., Vansteenwegen P., Lu K.  
  *Well-tuned algorithms for the Team Orienteering Problem with Time Windows.*  
  **Journal of the Operational Research Society**, 68, 861–876 (2017).

➡️ For detailed format descriptions and data availability, see the original publications above.

---

## 🎓 Academic Context

This solver and its underlying algorithms were developed as part of the Master's thesis:

**_“Das Team Orienteering Problem mit Zeitfenstern und Prioritätsklassen zur Evakuierung bei Starkregenkatastrophen”_**  

The work focuses on extending the classical Team Orienteering Problem with Time Windows (TOPTW) by incorporating **priority classes** and **evacuation-specific constraints**, addressing real-world challenges in **disaster management and emergency logistics**.


