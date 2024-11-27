import sys
from util.constants import MAXINT, SUCCESS
from util.utils import get_types




def build_surrogate(func_str='prf', config_space=None, rng=None, history_hpo_data=None, context=None):
    assert config_space is not None
    func_str = func_str.lower()
    types, bounds = get_types(config_space)
    seed = rng.randint(MAXINT)
    if func_str == 'prf':
        try:
            from transfer.surrogate.base.rf_with_instances import RandomForestWithInstances
            return RandomForestWithInstances(types=types, bounds=bounds, seed=seed, config_space=config_space)
        except ModuleNotFoundError:
            from transfer.surrogate.base.rf_with_instances_sklearn import skRandomForestWithInstances
            print('[Build Surrogate] Use probabilistic random forest based on scikit-learn. For better performance, '
                    'please install pyrfr: '
                    'https://open-box.readthedocs.io/en/latest/installation/install_pyrfr.html')
            return skRandomForestWithInstances(types=types, bounds=bounds, seed=seed)
    elif func_str == 'lightgbm':
        from transfer.surrogate.lightgbm import LightGBM
        return LightGBM(config_space, types=types, bounds=bounds, seed=seed)
    elif func_str.startswith('gp'):
        from transfer.surrogate.base.build_gp import create_gp_model
        return create_gp_model(model_type=func_str,
                               config_space=config_space,
                               types=types,
                               bounds=bounds,
                               rng=rng)

    


def surrogate_switch():
    pass