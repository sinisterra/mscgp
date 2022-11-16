# COVID-19 Data Analysis with Multi-objective Evolutionary Algorithm for Causal Association Rules Mining

Supplementary Material for article _COVID-19 Data Analysis with Multi-objective
Evolutionary Algorithm for Causal Association Rules Mining_ in Mathematical and
Computational Applications journal, Special Issue "Evolutionary Multi-objective
Optimization: An Honorary Issue Dedicated to Professor Kalyanmoy Deb"

Authors: Santiago Sinisterra-Sierra [1], Salvador Godoy-Calderón [1] and
Miriam Pescador-Rojas [2]

[1] Centro de Investigación en Computación, Instituto Politécnico Nacional, Ciudad
de México, México;
[2] Escuela Superior de Cómputo, Instituto Politécnico Nacional, Ciudad de México, México.

## Installation

Make sure to install Python 3.9 through `anaconda`. Then, run
`pip install -r requirements.txt` to install the required Python dependencies.

This implementation uses Clickhouse tables in a local instance. Check
https://clickhouse.com/docs/en/install/ for details on how to install
Clickhouse.

Rule results are cached in a `redis` database deployed with Docker. Please
install `docker` and `docker-compose`, then
`cd redis_cache && docker-compose up` to start Redis.

## Running the algorithm

Run `python main.py` to execute the algorithm. Each run creates a new folder
named with the timestamp of the execution. The file `selection.csv` contains the
results of the algorithm. For further exploration of more evaluation measures,
`finals.csv` contains all the possible evaluation measures considered by the
algorithm.

<!-- ## Contributions

Code by Santiago Sinisterra Sierra.

Centro de Investigación en Computación (CIC-IPN), Instituto Politécnico
Nacional, Mexico. -->
