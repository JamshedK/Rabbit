from ConfigSpace.util import deactivate_inactive_hyperparameters
import numpy as np
from scipy import stats
import numbers

from ConfigSpace import (
    UniformIntegerHyperparameter,
    UniformFloatHyperparameter,
    CategoricalHyperparameter,
    OrdinalHyperparameter,
    Configuration,
    ConfigurationSpace,
    Constant
)

# Transform 

def bilog_transform(X: np.ndarray):
    """Magnify the difference between X and 0"""
    X = X.copy()
    idx = (X >= 0)
    X[idx] = np.log(1 + X[idx])
    X[~idx] = -np.log(1 - X[~idx])
    return X

def gaussian_transform(X: np.ndarray):
    """
    Transform data into Gaussian by applying psi = Phi^{-1} o F where F is the truncated ECDF.
    References:
    [1] Andrew Gordon Wilson and Zoubin Ghahramani. Copula processes.
        In Proceedings of the 23rd International Conference on Neural Information Processing
        Systems - Volume 2, NIPS’10, pages 2460–2468, USA, 2010. Curran Associates Inc.
    [2] Salinas, D.; Shen, H.; and Perrone, V. 2020.
        A Quantile-based Approach for Hyperparameter Transfer Learning.
        In International conference on machine learning, 7706–7716.
    """
    if X.ndim == 2:
        z = np.hstack([
            gaussian_transform(x.reshape(-1)).reshape(-1, 1)
            for x in np.hsplit(X, X.shape[1])
        ])
        return z
    assert X.ndim == 1

    def winsorized_delta(n):
        return 1.0 / (4.0 * n ** 0.25 * np.sqrt(np.pi * np.log(n)))

    def truncated_quantile(X):
        idx = np.argsort(X)
        rank = np.argsort(idx)
        quantile = rank / (X.shape[0] - 1)
        delta = winsorized_delta(X.shape[0])
        return np.clip(quantile, a_min=delta, a_max=1 - delta)

    return stats.norm.ppf(truncated_quantile(X))


_func_dict = {
    'bilog': bilog_transform,
    'gaussian': gaussian_transform,
    None: lambda x: x,
}



def get_transform_function(transform: str):
    if transform in _func_dict.keys():
        return _func_dict[transform]
    else:
        raise ValueError('Invalid transform: %s' % (transform, ))
    

def check_random_state(seed):
    """ from [sklearn.utils.check_random_state]
    Turn seed into a np.random.RandomState instance

    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                    ' instance' % seed)

# Normalization
def zero_mean_unit_var_normalization(X, mean=None, std=None):
    if mean is None:
        mean = np.mean(X, axis=0)
    if std is None:
        std = np.std(X, axis=0) + 1e-4

    X_normalized = (X - mean) / std

    return X_normalized, mean, std

def zero_one_normalization(X, lower=None, upper=None):
    if lower is None:
        lower = np.min(X, axis=0)
    if upper is None:
        upper = np.max(X, axis=0)

    X_normalized = np.true_divide((X - lower), (upper - lower))

    return X_normalized, lower, upper


# ConfigSpace

def get_config_from_dict(config_dict: dict, config_space: ConfigurationSpace):
    config = deactivate_inactive_hyperparameters(configuration_space=config_space,
                                                 configuration=config_dict)
    return config

def convert_array_to_configurations(arr: np.ndarray, config_space: ConfigurationSpace, origin=None) -> list[Configuration]:
    """Convert a numpy array back to a list of SMAC Configuration objects.

    Parameters
    ----------
    arr : np.ndarray
        Array with configuration hyperparameters.
    config_space : ConfigurationSpace
        The configuration space to reconstruct the Configuration objects.

    Returns
    -------
    List[Configuration]
        List of Configuration objects reconstructed from the array.
    """
    configurations = []
    for row in arr:
        r = row.flatten().numpy()

        values = {}
        for index, hp in enumerate(config_space.get_hyperparameters()):
            if isinstance(hp, CategoricalHyperparameter):
                choices = hp.choices
                num_choices = len(choices)
                # Compute the index of the choice from the normalized value
                normalized_value = r[index]
                choice_index = int(round(normalized_value * (num_choices - 1)))
                # values[hp.name] = choices[choice_index]
                r[index] = choice_index
            # else:
            #     # Handle other types of hyperparameters if necessary
            #     values[hp.name] = None

        # config = Configuration(config_space, values, origin=origin)
        config = Configuration(config_space, vector=r, origin=origin)
        configurations.append(config)
    return configurations

def convert_configurations_to_array(configs: list[Configuration]) -> np.ndarray:
    """Impute inactive hyperparameters in configurations with their default.

    Necessary to apply an EPM to the data.

    Parameters
    ----------
    configs : List[Configuration]
        List of configuration objects.

    Returns
    -------
    np.ndarray
        Array with configuration hyperparameters. Inactive values are imputed
        with their default value.
    """
    configs_array = np.array([config.get_array() for config in configs],
                             dtype=np.float64)
    configuration_space = configs[0].config_space
    return impute_default_values(configuration_space, configs_array)


def impute_default_values(
        configuration_space: ConfigurationSpace,
        configs_array: np.ndarray
) -> np.ndarray:
    """Impute inactive hyperparameters in configuration array with their default.

    Necessary to apply an EPM to the data.

    Parameters
    ----------
    configuration_space : ConfigurationSpace
    
    configs_array : np.ndarray
        Array of configurations.

    Returns
    -------
    np.ndarray
        Array with configuration hyperparameters. Inactive values are imputed
        with their default value.
    """

    # categorical_hps = {hp.name: hp for hp in configuration_space.get_hyperparameters() if isinstance(hp, CategoricalHyperparameter)}
    for i, hp in enumerate(configuration_space.get_hyperparameters()):
        default = hp.default_value
        col = i  # Column index in configs_array
        if isinstance(hp, CategoricalHyperparameter):
            choices = hp.choices
            # choice_to_index = {choice: idx for idx, choice in enumerate(choices)}
            nonfinite_mask = ~np.isfinite(configs_array[:, col])
            configs_array[nonfinite_mask, col] = choices.index(default)
            if len(choices)>1:
                configs_array[:, col] = [v / (len(choices) - 1) for v in configs_array[:, col]]  
            else:
                configs_array[:, col] = 0
        else :
            nonfinite_mask = ~np.isfinite(configs_array[:, col])
            configs_array[nonfinite_mask, col] = default / (hp.upper_vectorized - hp.lower_vectorized)

    # print(configs_array)
    return configs_array


def get_types(config_space, instance_features=None):
    """TODO"""
    # Extract types vector for rf from config space and the bounds
    types = np.zeros(len(config_space.get_hyperparameters()),
                     dtype=np.uint)
    bounds = [(np.nan, np.nan)]*types.shape[0]

    for i, param in enumerate(config_space.get_hyperparameters()):
        if isinstance(param, (CategoricalHyperparameter)):
            n_cats = len(param.choices)
            types[i] = n_cats
            bounds[i] = (int(n_cats), np.nan)

        elif isinstance(param, (OrdinalHyperparameter)):
            n_cats = len(param.sequence)
            types[i] = 0
            bounds[i] = (0, int(n_cats) - 1)

        elif isinstance(param, Constant):
            # for constants we simply set types to 0
            # which makes it a numerical parameter
            types[i] = 0
            bounds[i] = (0, np.nan)
            # and we leave the bounds to be 0 for now
        elif isinstance(param, UniformFloatHyperparameter):         # Are sampled on the unit hypercube thus the bounds
            # bounds[i] = (float(param.lower), float(param.upper))  # are always 0.0, 1.0
            bounds[i] = (0.0, 1.0)
        elif isinstance(param, UniformIntegerHyperparameter):
            # bounds[i] = (int(param.lower), int(param.upper))
            bounds[i] = (0.0, 1.0)
        elif not isinstance(param, (UniformFloatHyperparameter,
                                    UniformIntegerHyperparameter,
                                    OrdinalHyperparameter)):
            raise TypeError("Unknown hyperparameter type %s" % type(param))
        
    if instance_features is not None:
        types = np.hstack(
            (types, np.zeros((instance_features.shape[1]))))

    types = np.array(types, dtype=np.uint)
    bounds = np.array(bounds, dtype=object)
    return types, bounds

import re
def transfer_unit(value):
    value = str(value)
    value = value.replace(" ", "")
    value = value.replace(",", "")
    if value.isalpha():
        value = "1" + value
    pattern = r'(\d+\.\d+|\d+)([a-zA-Z]+)'
    match = re.match(pattern, value)
    if not match:
        return float(value)
    number, unit = match.group(1), match.group(2)
    unit_to_size = {
        'kB': 1024 ** 1,
        'KB': 1024 ** 1,
        'MB': 1024 ** 2,
        'GB': 1024 ** 3,
        'TB': 1024 ** 4,
        'K': 1024 ** 1,
        'M': 1024 ** 2,
        'G': 1024 ** 3,
        'B': 1,
        'us': 0.001,
        'ms': 1,
        's': 1000,
        'min': 60000,
        'h': 60 * 60000,
        'day': 24 * 60 * 60000,
    }
    return float(number) * unit_to_size[unit]
    
def type_transfer(knob_type, value):
    value = str(value)
    value = value.replace(",", "")
    if knob_type == "integer":
        # return int(round(float(value)))
        try:
            return int(value)
        except:
            return int(round(float(value)))
    if knob_type == "real":
        return float(value)
