from space_optimizer.init_space import InitSpace
import os
import json
import numpy as np
import time
from util.logger_config import logger
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
import gpytorch
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import UpperConfidenceBound, ExpectedImprovement, qKnowledgeGradient, LogExpectedImprovement
from botorch.optim import optimize_acqf
from util.lhs import sample_lhs
from smac.runhistory.runhistory import RunHistory
from util.utils import convert_configurations_to_array, convert_array_to_configurations
from ConfigSpace import CategoricalHyperparameter, UniformIntegerHyperparameter, UniformFloatHyperparameter


class GPStage(InitSpace):

    def __init__(self, dbms, test, timeout, target_knobs_path, objective, dir_path, seed):
        super().__init__(dbms, test, timeout, target_knobs_path, seed)
        self.method = "GP"
        self.objective = objective
        self.runhistory = RunHistory()
        self.dir_path = dir_path
        if not os.path.exists(self.dir_path):
            os.makedirs(self.dir_path)
        self.runhistory_path = os.path.join(self.dir_path, 'runhistory.json')
        self.intensifier = []
        self._save_configspace()

    def _save_configspace(self):
            configspace_path = os.path.join(self.dir_path, 'configspace.json')
            
            config_list = []
            for hp in self.search_space.get_hyperparameters():
                if isinstance(hp, CategoricalHyperparameter):
                    config_list.append(
                    {
                        "name": hp.name,
                        "type": hp.__class__.__name__,
                        "choices": hp.choices
                    })
                elif isinstance(hp, UniformIntegerHyperparameter):
                    config_list.append(
                        {
                            "name": hp.name,
                            "type": "uniform_int",   # 根据不同类型的超参数填写
                            "log": hp.log,   # 是否是 log-scale
                            "lower": hp.lower,  # 下限
                            "upper": hp.upper,  # 上限
                            "default": hp.default_value,  # 默认值
                            "q": None       # 离散间隔，对于整数超参数
                        }
                    )
                elif isinstance(hp, UniformFloatHyperparameter):
                    config_list.append(
                        {
                            "name": hp.name,
                            "type": "uniform_float",   # 根据不同类型的超参数填写
                            "log": hp.log,   # 是否是 log-scale
                            "lower": hp.lower,  # 下限
                            "upper": hp.upper,  # 上限
                            "default": hp.default_value,  # 默认值
                            "q": None       # 离散间隔，对于整数超参数
                        }
                    )
            # 保存为 JSON 文件
            config_dict = {
                "hyperparameters": config_list
            }
            with open(configspace_path, 'w') as f:
                json.dump(config_dict, f,indent=4)

    def _save_json(self):
        with open(self.runhistory_path, "r") as f:
            data = json.load(f)
            data = data["data"]

        trajectory = []
        max_cost = data[0][4]
        trajectory.append({"config_ids":[data[0][0]], "costs":[max_cost], "trial":data[0][0], "walltime": data[0][5]})
        for d in data:
            if d[4] < max_cost:
                trajectory.append({"config_ids":[d[0]], "costs":[d[4]], "trial":d[0], "walltime": d[5]})
                max_cost = d[4]

        intensifier_path = os.path.join(self.dir_path, 'intensifier.json')
        with open(intensifier_path, "w") as f:
            json.dump({"trajectory":trajectory}, f, indent=2)

    def lhs(self, lhs_num):
        if lhs_num == 0:
            return []
        lhs_samples = sample_lhs(self.search_space, lhs_num)
        default_config = self.search_space.get_default_configuration()
        initial_configs = [default_config] + lhs_samples
        
        # 返回一个Configuration的列表
        return initial_configs

    def _get_next_point(self):
        train_X, train_Y = [], []
        configs = self.runhistory.get_configs('cost')
        train_X = convert_configurations_to_array(configs)
        costs = []
        context = []
        for config in configs:
            cost = self.runhistory.get_cost(config)
            costs.append([cost])
            context.append({'configuration': config, 'cost': cost})
        train_Y= np.array(costs, dtype=np.float64).reshape(-1, 1)
        train_Y_mean = np.mean(train_Y)
        train_Y_std = np.std(train_Y)
        train_Y_standardized = (train_Y - train_Y_mean) / train_Y_std

        X = torch.tensor(train_X, dtype=torch.float64)
        Y = torch.tensor(train_Y_standardized, dtype=torch.float64)

        gp = SingleTaskGP(X, Y)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        print("best_f", torch.min(Y).item())
        # UCB = UpperConfidenceBound(gp, beta=0.1, maximize=False if self.objective == 'lat' else True)
        EI = LogExpectedImprovement(gp, best_f=torch.min(Y).item(), maximize=False)
        bounds = torch.stack([torch.zeros(self.target_knobs_num), torch.ones(self.target_knobs_num)])
        with gpytorch.settings.cholesky_jitter(1e-1):
            candidate, acq_value = optimize_acqf(
                EI, bounds=bounds, q=1, num_restarts=5, raw_samples=2000, return_best_only=False
            )
        idx = int(acq_value.argmax())
        config_gp = convert_array_to_configurations(candidate[idx], self.search_space, origin="Config Selected From GP")
        
        noise_variance = gp.likelihood.noise.item()  # 获取噪声方差
        sigma_epsilon = noise_variance ** 0.5  # 噪声标准差
        max_ig = -np.inf
        # 计算IG
        for X_c in candidate:
            K = gp.covar_module(X_c, X_c).evaluate()

            ig = 0.5 * torch.logdet(torch.eye(len(X_c)) + sigma_epsilon**(-2) * K).item()
            if ig > max_ig:
                max_ig = ig
        logger.info(max_ig)

        with open('mig_data_gp.txt', 'a') as f:
            f.write(f"MIG = {max_ig}\n")
        
        return config_gp[0]
    
    def _set_and_add_history(self, config):
        tps, lat, t = self.set_and_replay_tps_and_lat(config)
        if self.objective == '-lat':
            cost = lat
        else:
            cost = -tps
        self.runhistory.add(config=config, cost=cost, time=t)
        self.runhistory.save(self.runhistory_path)
    
    def tune(self, trials_number, warm_start_times):
        knobs_set = self.lhs(warm_start_times)
        logger.info("warm start begin!!!")
        for knobs in knobs_set:
            self._set_and_add_history(knobs)
        logger.info("warm start over!!!")
        for _ in range(trials_number - warm_start_times):
            now = time.time()
            knobs = self._get_next_point()
            logger.info(f"recommend next knobs spent {time.time() - now}s")
            self._set_and_add_history(knobs)
