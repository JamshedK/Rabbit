import numpy as np
from scipy.stats import qmc
import ConfigSpace as CS



# 使用拉丁超立方体采样
def sample_lhs(config_space, num_samples):
    # 获取超参数的离散值
    param_names = [param.name for param in config_space.get_hyperparameters()]
    # values = [param.choices for param in config_space.get_hyperparameters()]
    num_params = len(param_names)
    
    # 使用 LHS 进行采样
    sampler = qmc.LatinHypercube(d=num_params)
    lhs_samples = sampler.random(n=num_samples)
    
    # 将 LHS 样本离散化为 Configuration 对象
    configurations = []
    # for sample in lhs_samples:
    #     config_dict = {}
    #     for i, param in enumerate(values):
    #         # 将样本值映射到离散值
    #         index = int(sample[i] * len(param))  # 取整
    #         config_dict[param_names[i]] = param[index]
        
    #     # 创建 Configuration 对象
    #     configuration = CS.Configuration(config_space, config_dict)
    #     configurations.append(configuration)
    for sample in lhs_samples:
        config_dict = {}
        for i, param in enumerate(config_space.get_hyperparameters()):
            if isinstance(param, CS.UniformIntegerHyperparameter):
                # 对整数类型参数进行处理
                config_dict[param_names[i]] = int(param.lower + sample[i] * (param.upper - param.lower))
            elif isinstance(param, CS.UniformFloatHyperparameter):
                # 对浮点数类型参数进行处理
                config_dict[param_names[i]] = param.lower + sample[i] * (param.upper - param.lower)
            else:
                # 处理离散值参数
                values = param.choices
                index = int(sample[i] * len(values))  # 取整
                config_dict[param_names[i]] = values[index]
        
        # 创建 Configuration 对象
        configuration = CS.Configuration(config_space, config_dict)
        configurations.append(configuration)
    
    
    return configurations

