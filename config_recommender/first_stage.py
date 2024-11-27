from space_optimizer.coarse_space import CoarseSpace
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

class FirstStage(CoarseSpace):
    def __init__(self, dbms, test, timeout, target_knobs_path, skill_path, incumbents_transfer_path, extra_knobs_configs_path, objective, dir_path, seed):
        super().__init__(dbms, test, timeout, target_knobs_path, skill_path, incumbents_transfer_path, extra_knobs_configs_path, seed)
        self.method = "LLM"
        self.objective = objective
        self.perfs = {}
        self.perfs['cur_cost'], self.perfs['default_cost'], self.perfs['best_cost'], self.perfs["last_best_cost"] = None, None, None, None
        self.runhistory = RunHistory()
        self.dir_path = dir_path
        # 如果目录不存在，则创建它
        if not os.path.exists(self.dir_path):
            os.makedirs(self.dir_path)
        self.runhistory_path = os.path.join(self.dir_path, 'runhistory.json')
        self.intensifier = [] 
        self.mig = []
        self._save_configspace()
    def _save_configspace(self):
        configspace_path = os.path.join(self.dir_path, 'configspace.json')
        config_dict = {
            "hyperparameters": [
                {
                    "name": hp.name,
                    "type": hp.__class__.__name__,
                    "choices": hp.choices
                } for hp in self.search_space.get_hyperparameters()
            ]
        }
        # 保存为 JSON 文件
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

    def _load_history_from_first(self, first_path):
        with open(first_path, "r") as json_file:
            data = json.load(json_file)
        costs = []
        for i in range(len(data["data"])):
            costs.append(data["data"][i][4])
        index_min_pairs = sorted(enumerate(costs), key=lambda x: x[1])
        # no ordering
        for index, value in index_min_pairs:
            config_id = index + 1
            config_value_dict = data["configs"][str(config_id)]
            config_cost = data["data"][index][4]
            assert value == config_cost

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
    
    def _check_sample_in_search_space(self, sample):
    # 遍历 sample 中的每一个键值对

        adjusted_config = {}
        for param_name, param_value in sample.items():

            hp = self.search_space.get_hyperparameter(param_name)
            
            # 检查值是否在合法范围内
            if not hp.is_legal(param_value):
                logger.info(f"参数 {param_name} 的值 {param_value} 不合法。")
                  # 如果是离散超参数
                  # TODO
                if hasattr(hp, 'choices'):
                    legal_values = list(hp.choices)
                    # 查找最近的合法值
                    closest_value = min(legal_values, key=lambda x: abs(float(x) - float(param_value)))
                    adjusted_config[param_name] = closest_value
            else:
                adjusted_config[param_name] = param_value

        # 如果所有参数都合法
        return adjusted_config

    def _get_next_point_hybrid(self, candidate_nums=5):
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
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        train_Y_standardized = scaler.fit_transform(train_Y)
        X = torch.tensor(train_X, dtype=torch.float64)
        Y = torch.tensor(train_Y_standardized, dtype=torch.float64)
        target = float(train_Y.min()) + 0.2 * (float(train_Y.max() - train_Y.min()))
        logger.info(target)
        gp = SingleTaskGP(X, Y)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        UCB = UpperConfidenceBound(gp, beta=0.1, maximize=False )
        EI = LogExpectedImprovement(gp, best_f=target, maximize=False )
        bounds = torch.stack([torch.zeros(self.target_knobs_num), torch.ones(self.target_knobs_num)])
        with gpytorch.settings.cholesky_jitter(1e-1):
            candidates_default, acq_values_default = optimize_acqf(
                EI, bounds=bounds, q=1, num_restarts=candidate_nums, raw_samples=2000, return_best_only=False
            )

        noise_variance = gp.likelihood.noise.item()  # 获取噪声方差
        sigma_epsilon = noise_variance ** 0.5  # 噪声标准差
        max_ig = -np.inf
        # 计算IG
        for X_c in candidates_default:
            K = gp.covar_module(X_c, X_c).evaluate()

            ig = 0.5 * torch.logdet(torch.eye(len(X_c)) + sigma_epsilon**(-2) * K).item()
            if ig > max_ig:
                max_ig = ig
        self.mig.append(max_ig)
        with open(os.path.join(self.dir_path, 'mig_data3.txt'), 'a') as f:
            f.write(f"MIG = {max_ig}\n")

        llm = LLM(self.dbms, self.test, None, context, self.objective, self.search_space)
        llm_initial_samples = llm.gen_candidates_llm(candidate_nums, target)
        configurations = []
        for sample in llm_initial_samples:
            # samples.append(transform_knobs2vector(self.dbms.knob_info, sample))
            sample = self._check_sample_in_search_space(sample)
            configuration = Configuration(self.search_space, sample, origin=" Config Selected From LLM")
            configurations.append(configuration)
        if configurations:
            samples = convert_configurations_to_array(configurations)
            samples = torch.tensor(samples, dtype=torch.float64)
            samples = samples.reshape(candidate_nums, 1, self.target_knobs_num)
            with gpytorch.settings.cholesky_jitter(1e-1):
                candidates_llm, acq_values_llm = optimize_acqf(
                    EI, bounds=bounds, q=1, num_restarts=candidate_nums, batch_initial_conditions=samples, return_best_only=False
                )
            candidates = torch.concat([candidates_default, candidates_llm], dim=0)
            acq_values = torch.concat([acq_values_default, acq_values_llm], dim=0)
        else:
            candidates = candidates_default
            acq_values = acq_values_default

        idx = int(acq_values.argmax())

        candidate_configs_default = convert_array_to_configurations(candidates_default, self.search_space, origin="Config Selected From LLM")
        candidate_configs = candidate_configs_default + configurations

        config_gp = convert_array_to_configurations(candidates[idx], self.search_space, origin="Config Selected From GP")

        return config_gp[0], candidate_configs, llm, target
    
    def log_expected_improvement(self, mu, sigma, best_y):
        # 计算 LogEI，假设我们希望最小化目标
        if sigma == 0:  # 避免除以零
            return 0.0
        Z = (best_y - mu) / sigma  # 注意这里反转了
        ei = (best_y - mu) * norm.cdf(Z) + sigma * norm.pdf(Z)
        
        # 计算对数期望改进
        log_ei = np.log(ei) if ei > 0 else -np.inf  # 避免对数零
        return log_ei

    def _get_next_point_llm(self, candidate_configs, llm, best_f):
        # 获得下一个候选的采样点
        # 获得均值和方差
        k = 5
    
        log_ei_values = []
        for config in candidate_configs:
            weights = []
            perfs = []
            for i in range(k):
                ans = llm.prediction(config)
                if ans == None:
                    continue
                perf = ans["Performance"]
                confidence = ans["Confidence"]
                perfs.append(perf)
                weights.append(confidence)
            weights = np.array(weights)
            perfs = np.array(perfs)
            
            # 计算均值
            mu = np.sum(weights * perfs) / np.sum(weights)
            # 计算标准差
            variance = np.sum(weights * (perfs - mu)**2) / np.sum(weights)
            sigma = np.sqrt(variance)
            log_ei = self.log_expected_improvement(mu, sigma, best_f)
            log_ei_values.append(log_ei)
        # 找到最大 LogEI 的配置
        best_index = np.argmax(log_ei_values)
        best_config = candidate_configs[best_index]
        return best_config

    def _get_reward(self):
        perf_first = self.perfs['last_best_cost']
        perf_last = self.perfs['cur_cost']
        if perf_first - perf_last > 0:
            return 1
        else:
            return 0
            
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

    def tune_end2end(self, trials_number, initial_config_number):
        # init
        config_set = self.lhs(initial_config_number)
        logger.info("warm start begin!!!")
        for config in config_set:
            self._set_and_add_history(config)
        logger.info("warm start over!!!")


        # trials
        patience_counter = 0
        patience_counter_max = 0
        number = initial_config_number
        self.cost_path = os.path.join(self.dir_path, 'cost.txt')
        for index in range(trials_number - initial_config_number):
        # for i in range(trials_number):
            number += 1
            now = time.time()
            config_gp, candidate_configs, llm, best_f = self._get_next_point_hybrid()
            logger.info(f"recommend next knobs spent {time.time() - now}s")
            
            config_llm = self._get_next_point_llm(candidate_configs, llm, best_f)
            logger.info(dict(config_llm))
            self._set_and_add_history(config_llm)


            with open(self.cost_path, "a") as f:
                f.write(f"Inter {index}: input_token({llm.input_token}; output_token({llm.output_token}; total_token({llm.token};)")



            if index > 0:
                mig_diff = abs(self.mig[index] - self.mig[index - 1])
                if mig_diff < 1e-4:
                    patience_counter +=1
                else:
                    patience_counter = 0

                if patience_counter >= 3:
                    logger.info("Terminating optimization due to low information gain.")
                    break
            if index > 0:
                mig_diff = abs(self.mig[index] - self.mig[index - 1])
                mig_max_diff = self.mig[index] - max(self.mig)
                if mig_diff < 1e-4 :
                    patience_counter +=1
                else:
                    patience_counter = 0
                
                if mig_max_diff < 0:
                    patience_counter_max +=1
                else:
                    patience_counter_max = 0

                if index >15 and (patience_counter >= 3 or patience_counter_max >=5):
                    logger.info("Terminating optimization due to low information gain.")
                    break 
        self._save_json()
        return number

