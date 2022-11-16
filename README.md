# Causal Rule Miner

Written by Santiago Sinisterra Sierra.

Centro de Investigación en Computación (CIC-IPN), Instituto Politécnico
Nacional, Mexico.

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
