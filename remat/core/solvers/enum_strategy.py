from enum import Enum


class SolveStrategy(Enum):
    NOT_SPECIFIED = 'NOT_SPECIFIED'
    CHEN_SQRTN = 'CHEN_SQRTN'
    CHEN_GREEDY = 'CHEN_GREEDY'
    CHEN_SQRTN_NOAP = 'CHEN_SQRTN_NOAP'
    CHEN_GREEDY_NOAP = 'CHEN_GREEDY_NOAP'
    OPTIMAL_ILP_GC = 'OPTIMAL_ILP_GC'
    CHECKPOINT_LAST_NODE = 'CHECKPOINT_LAST_NODE'
    CHECKPOINT_ALL = 'CHECKPOINT_ALL'
    CHECKPOINT_ALL_AP = 'CHECKPOINT_ALL_AP'
    GRIEWANK_LOGN = 'GRIEWANK_LOGN'

    @classmethod
    def get_description(cls, val, model_name=None):
        is_linear = model_name in ("VGG16", "VGG19", "MobileNet")
        return {
            cls.CHEN_SQRTN: "AP $\\sqrt{n}$",
            cls.CHEN_GREEDY: "AP greedy",
            cls.CHEN_SQRTN_NOAP: "Generalized $\\sqrt{n}$" if not is_linear else "Chen et al. $\\sqrt{n}$",
            cls.CHEN_GREEDY_NOAP: "Generalized greedy",
            cls.OPTIMAL_ILP_GC: "Optimal MILP (proposed)",
            cls.CHECKPOINT_LAST_NODE: "Checkpoint last node",
            cls.CHECKPOINT_ALL: "Checkpoint all (ideal)",
            cls.CHECKPOINT_ALL_AP: "Checkpoint all APs",
            cls.GRIEWANK_LOGN: "Griewank et al. $\\log~n$" if is_linear else "AP $\\log~n$",
        }[val]

    # todo move this to experiments codebase
    @classmethod
    def get_plot_params(cls, val):
        from matplotlib import rcParams
        fullsize = rcParams["lines.markersize"]
        halfsize = fullsize / 2
        bigger = fullsize * 1.5
        mapping = {
            cls.CHEN_SQRTN: ("c", "D", halfsize),
            cls.CHEN_SQRTN_NOAP: ("c", "^", halfsize),
            cls.CHEN_GREEDY: ("g", ".", fullsize),
            cls.CHEN_GREEDY_NOAP: ("g", "+", fullsize),
            cls.OPTIMAL_ILP_GC: ("r", "s", halfsize),
            cls.CHECKPOINT_ALL: ("k", "*", bigger),
            cls.CHECKPOINT_ALL_AP: ("b", "x", fullsize),
            cls.GRIEWANK_LOGN: ("m", "p", fullsize),
        }
        if val in mapping:
            return mapping[val]
        raise NotImplementedError(f"No plotting parameters for strategy {val}")

    @classmethod
    def get_version(cls, val):
        return {
            cls.CHEN_SQRTN: "v1.1",
            cls.CHEN_SQRTN_NOAP: "v1.1",
            cls.CHEN_GREEDY: "v1.1",
            cls.CHEN_GREEDY_NOAP: "v1.1",
            cls.OPTIMAL_ILP_GC: "v3",
            cls.CHECKPOINT_LAST_NODE: "v1.1",
            cls.CHECKPOINT_ALL: "v1.1",
            cls.CHECKPOINT_ALL_AP: "v1.1",
            cls.GRIEWANK_LOGN: "v1.3",  # 1.3 -> fix AP point mapping
        }[val]
