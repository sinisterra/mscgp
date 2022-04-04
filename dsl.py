from dataclasses import dataclass
from typing import Tuple, List, Any, Set, Dict
import pandas as pd


@dataclass(frozen=True)
class Context:
    id: int
    exec_run_path: str
    antecedent: Tuple[int, int]
    consequent: Tuple[int, int]
    covariates: Tuple[str, ...]
    dataframe: str
    omit: Tuple[str]
    pop_size: int
    stop_condition: Tuple[str, int]
    measures: Tuple[str, ...]
    optimize: Tuple[str, ...]
    groups: Tuple[Tuple[str, ...], Tuple[str, ...]]
    use_groups: bool
    selector_restrictions: Tuple[Tuple[str, ...], Tuple[str, ...]]
    cover_mode: str
    aptitude_fn: Tuple[str,...]


@dataclass(frozen=True)
class ConfusionMatrix:
    tp: int
    fp: int
    fn: int
    tn: int
    total: int
    exp_size: int
    control_size: int
    norm_factor: str
    used_factor: int


Selector = Tuple[str, Any]


@dataclass
class AlgorithmState:
    generation: int
    evaluations: int
    selectors: Dict[str, List[Selector]]
    attributes: List[str]
    population: pd.DataFrame
    history: List[Any]  # AlgorithmState
    convergence: pd.DataFrame
    pop_history: pd.DataFrame
    elites: pd.DataFrame
