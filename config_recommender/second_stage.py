from space_optimizer.fine_space import FineSpace
import os
import json
import numpy as np
import re
import time
from util.logger_config import logger
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from scipy.stats import norm
import gpytorch
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import UpperConfidenceBound, ExpectedImprovement, qKnowledgeGradient, LogExpectedImprovement
from botorch.optim import optimize_acqf
from ConfigSpace import Configuration, ConfigurationSpace
from util.lhs import sample_lhs
from smac.runhistory.runhistory import RunHistory
from util.utils import convert_configurations_to_array, convert_array_to_configurations
from util.transform import transform_vector2knobs, transform_knobs2vector
from RAG.llm import LLM
from ConfigSpace import CategoricalHyperparameter, UniformIntegerHyperparameter, UniformFloatHyperparameter
import sys
import random
from util.kernels import NewKernel
import botorch

# GP 模型定义
class GPModel(gpytorch.models.ExactGP, botorch.models.gpytorch.GPyTorchModel):
    num_outputs = 1
    def __init__(self, train_x, train_y, likelihood, kernel=None):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        # self.covar_module = gpytorch.kernels.MaternKernel(2.5)
        # self.covar_module = kernel
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(
                nu=2.5,
                ard_num_dims=None    
            )
        )


    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)  # 计算协方差矩阵
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class SecondStage(FineSpace):
    def __init__(self, dbms, test, timeout, target_knobs_path, skill_path, incumbents_transfer_path, extra_knobs_configs_path, objective, dir_path, extra_knob_space, seed):
        super().__init__(dbms, test, timeout, target_knobs_path, skill_path, incumbents_transfer_path, extra_knobs_configs_path, seed)
        self.method = "LLM"
        self.objective = objective
        self.perfs = {}
        self.perfs['cur_cost'], self.perfs['default_cost'], self.perfs['best_cost'], self.perfs["last_best_cost"] = None, None, None, None
        self.runhistory = RunHistory()
        self.dir_path = dir_path
        self.runhistory_path = os.path.join(self.dir_path, 'runhistory.json')
        self.first_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(self.dir_path))), f'first/{seed}/runhistory.json' )
        self.extra_knobs_space_path = extra_knob_space
        self.intensifier = []  # TODO
        self._load_extra_knobs()
        self._save_configspace()

    def _load_extra_knobs(self):
        # 如果目录不存在，则创建它
        if not os.path.exists(self.dir_path):
            os.makedirs(self.dir_path)

        base, ext = os.path.splitext(self.target_knobs_path)
        target_knobs_all_path = f"{base}_all_sorted{ext}"
        with open(target_knobs_all_path, "r") as f:
            data = f.readlines()
            self.target_knobs_all = [ d.strip() for d in data]
            self.target_knobs_add = [ knob for knob in self.target_knobs_all if knob not in self.target_knobs]

        with open(self.extra_knobs_space_path, "r") as f:
            self.extra_knobs_space = json.load(f)

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

    def _get_next_point(self, candidate_nums=5):
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

        X = torch.tensor(train_X, dtype=torch.float32)
        Y = torch.tensor(train_Y_standardized, dtype=torch.float32)
        Y = Y.squeeze(-1)
        target = float(train_Y_standardized.min())
        logger.info(target)
        botorch.settings.debug(True)
        noise = 1e-3
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(dtype=torch.float)
        gp = GPModel(X, Y, likelihood=likelihood).to(dtype=torch.float)
        gp.likelihood.noise = noise
        mll = ExactMarginalLogLikelihood(likelihood, gp).to(dtype=torch.float)

        # 优化
        optimizer = torch.optim.Adam(gp.parameters(), lr=0.01)
        gp.train()
        likelihood.train()

        losses = []
        for i in range(1000):
            optimizer.zero_grad()
            output = gp(X)
            loss = -mll(output, Y)
            loss.backward()
            losses.append(loss.item())
            optimizer.step()

        # 切换到评估模式
        gp.eval()
        likelihood.eval()

        # UCB = UpperConfidenceBound(gp, beta=0.1, maximize=False ).to(dtype=torch.float)
        best_f = torch.min(Y).item()

        EI = LogExpectedImprovement(gp, best_f=best_f, maximize=False).to(dtype=torch.float)
        bounds = torch.stack([torch.zeros(self.target_knobs_num, dtype=torch.float), torch.ones(self.target_knobs_num, dtype=torch.float)])
        with gpytorch.settings.cholesky_jitter(1e-1):
            candidates_default, acq_values_default = optimize_acqf(
                EI, bounds=bounds, q=1, num_restarts=candidate_nums, raw_samples=2000, return_best_only=False
            )
        idx = int(acq_values_default.argmax())
        config_gp = convert_array_to_configurations(candidates_default[idx], self.search_space, origin="Config Selected From GP")
        return config_gp[0]
    
            
    def lhs(self, lhs_num):
        if lhs_num == 0:
            return []
        lhs_samples = sample_lhs(self.search_space, lhs_num)
        default_config = self.search_space.get_default_configuration()
        initial_configs = [default_config] + lhs_samples
        
        # 返回一个Configuration的列表
        return initial_configs
    
    
    def _set_and_add_history(self, config):
        tps, lat, t = self.set_and_replay_tps_and_lat(config)
        if self.objective == '-lat':
            cost = lat
        else:
            cost = -tps
        self.runhistory.add(config=config, cost=cost, time=t)
        self.runhistory.save(self.runhistory_path)
        if not self.perfs['cur_cost']:
            self.perfs['cur_cost'] = cost
            self.perfs['default_cost'] = cost
            self.perfs['best_cost'] = cost
            self.perfs['last_best_cost'] = cost
        else:
            self.perfs['cur_cost'] = cost
            if self.perfs['best_cost'] > cost:
                self.perfs['last_best_cost'] = self.perfs['best_cost']
                self.perfs['best_cost'] = cost

    def _load_history_from_first(self):
        with open(self.first_path, "r") as json_file:
            data = json.load(json_file)
        costs = []
        for i in range(len(data["data"])):
            costs.append(data["data"][i][4])
        index_min_pairs = sorted(enumerate(costs), key=lambda x: x[1])

        for index, value in index_min_pairs:
            config_id = index + 1
            config_value_dict = data["configs"][str(config_id)]
            config_cost = data["data"][index][4]
            assert value == config_cost
            # make type transformation from coarse to fine 
            transfer_config_value_dict = {}
            for key, value in config_value_dict.items():
                hp = self.search_space[key]
                if isinstance(hp, CategoricalHyperparameter):
                    transfer_config_value_dict[key] = str(value)
                elif isinstance(hp, UniformIntegerHyperparameter):
                    transfer_config_value_dict[key] = int(value) 
                elif isinstance(hp, UniformFloatHyperparameter):
                    transfer_config_value_dict[key] = float(value)
                else:
                    transfer_config_value_dict[key] = value
            config = Configuration(self.search_space, transfer_config_value_dict)
            self.runhistory.add(config, config_cost, seed=self.seed)
    
    def change_search_space(self, knobs):

        for knob in knobs:
            if knob in list(self.search_space.keys()):
                logger.info(f"旋钮{knob} 已经在搜索空间中")
                continue
            logger.info(f"Add search space for knob: {knob}")
            info = self.dbms.knob_info[knob]
            knob_type = info["vartype"]
            boot_value = info["reset_val"]
            unit = info["unit"]
            max_value = info["max_val"]
            min_value = info["min_val"]
            suggested_values_origin = self.extra_knobs_space[knob]

            if knob_type == "enum":
                choices = [item.upper() for item in info["enumvals"]]
                suggested_values = [item.upper() for item in suggested_values_origin if item.upper() in choices]

            elif knob_type == "bool":
                suggested_values = [item.upper() for item in suggested_values_origin if item.upper() in ["ON","OFF"]]
            
            else:
                if unit is not None:
                    unit = self._transfer_unit(unit)
                    suggested_values_origin = [(self._transfer_unit(value) / unit) for value in suggested_values_origin]
                try:
                    suggested_values_origin = [self._type_transfer(knob_type, value) for value in suggested_values_origin]
                    min_value = self._type_transfer(knob_type, min_value)
                    max_value = self._type_transfer(knob_type, max_value)
                    boot_value = self._type_transfer(knob_type, boot_value)
                except:

                    def match_num(value):
                        pattern = r"(\d+)"
                        match = re.match(pattern, value)
                        if match:
                            return match.group(1)
                        else:
                            return ""
                    pattern = r"(\d+)"
                    suggested_values_origin = [self._type_transfer(knob_type, re.match(pattern, value).group(1)) for value in suggested_values_origin if re.match(pattern, value) is not None]
                    min_value = self._type_transfer(knob_type, match_num(min_value))
                    max_value = self._type_transfer(knob_type, match_num(max_value))
                    boot_value = self._type_transfer(knob_type, match_num(boot_value))
                
                suggested_values = [item for item in suggested_values_origin if item < max_value and item > min_value]

            if boot_value not in suggested_values:
                suggested_values.append(boot_value)
            suggested_values = list(set(suggested_values))
            new_knob = CategoricalHyperparameter(
                    knob,
                    [str(value) for value in suggested_values],
                    default_value = str(boot_value),
                )
            self.search_space.add(new_knob)

    def select_add_knobs(self, epsilon=0.2, num_selection=3):
        # 按照epsilon_greedy
        selected_knobs = []
        for _ in range(num_selection):
            if len(self.target_knobs_add)==1:
                selected_knob = self.target_knobs_add[0] 
                selected_knobs.append(selected_knob)
                self.target_knobs_add.remove(selected_knob)
                self.target_knobs.append(selected_knob)
                self.target_knobs_num +=1
            elif len(self.target_knobs_add)>1:
                if random.random() < epsilon :
                    selected_knob = random.choice(self.target_knobs_add[1:])
                else:
                    selected_knob = self.target_knobs_add[0] 
                selected_knobs.append(selected_knob)
                self.target_knobs_add.remove(selected_knob)
                self.target_knobs.append(selected_knob)
                self.target_knobs_num +=1
            else:
                logger.info("add knobs over")
                break

        logger.info(selected_knobs)
        return selected_knobs
    
    def extend_runhistory(self, configs, costs,  new_config_space):
    
        runhistory = RunHistory()
        for index, config in enumerate(configs):
            # 创建一个新的 Configuration，基于新的 config space
            new_config_dict = {}
            # 将旧配置空间的参数值拷贝到新的配置空间中
            names = []
            config_value = config._values
            for name, value in config_value.items():
                names.append(name)
            for hp in new_config_space.get_hyperparameters():
                if hp.name not in names:
                    # 如果在旧的配置中缺少此超参数，使用默认值填充
                    new_config_dict[hp.name] = hp.default_value
                else:
                    new_config_dict[hp.name] = config[hp.name]
            
            # 创建新的 Configuration 对象
            new_config = Configuration(new_config_space, values=new_config_dict)
            
            # 获取旧配置的性能并将其与新配置一起存储

            cost = costs[index]
            runhistory.add(new_config, cost)
        
        return runhistory

    def tune_end2end(self, trials_number):
        # init
        logger.info("load history begin!!!")
        self._load_history_from_first()
        logger.info("load history over!!!")
        
        # trials
        for i in range(trials_number):
            now = time.time()

            # change history
            # change self.search_space and change self.runhistory, self.target_knobs_num
            if i%4==0 and len(self.target_knobs_add)>0:
                knob_list = self.select_add_knobs()
                configs = self.runhistory.get_configs()
                costs = [self.runhistory.get_cost(config) for config in configs]
                self.change_search_space(knob_list)
                self._save_configspace()
                self.runhistory = None
                self.runhistory = self.extend_runhistory(configs, costs, self.search_space)

            config_gp = self._get_next_point()
            logger.info(f"recommend next knobs spent {time.time() - now}s")
            logger.info(dict(config_gp))
            self._set_and_add_history(config_gp)

        self._save_json()

