# License: MIT
import os
import pdb
import sys
import time
import json
import collections
from typing import List, Union
import numpy as np
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter, Constant, \
    OrdinalHyperparameter
from ConfigSpace import Configuration, ConfigurationSpace
from util.constants import MAXINT, SUCCESS
from util.logger_config import logger
from util.utils import get_config_from_dict
from util.utils import  get_transform_function
from util.utils import convert_configurations_to_array

Perf = collections.namedtuple(
    'perf', ['cost', 'time', 'status', 'additional_info'])

Observation = collections.namedtuple(
    'Observation', ['config', 'trial_state', 'constraints', 'objs', 'elapsed_time',  'iter_time','EM', 'IM', 'resource', 'info', 'context'])


def detect_valid_history_file(dir):
    if not os.path.exists(dir):
        return []
    files = os.listdir(dir)
    valid_files = []
    for fn in files:
        try:
            with open(fn) as fp:
                all_data = json.load(fp)
        except Exception as e:
            continue
        data = all_data['data']
        valid_count = 0
        for item in data:
            if item['trial_state'] == 0:
                valid_count = valid_count + 1
            if valid_count > len(data)/2:
                valid_files.append(files)
                continue
    return valid_files


def load_history_from_filelist(task_id, fileL, config_space):

    data_mutipleL = list()
    for fn in fileL:
        try:
            with open(fn) as fp:
                all_data = json.load(fp)
        except Exception as e:
            print('Encountered exception %s while reading runhistory from %s. '
                  'Not adding any runs!', e, fn, )
            return

        info = all_data["info"]
        data = all_data["data"]
        data_mutipleL = data_mutipleL + data


    file_out = 'history_{}.json'.format(task_id)
    with open(file_out, "w") as fp:
        json.dump({"info": info, "data": data_mutipleL}, fp, indent=2)

    history_container = HistoryContainer(task_id, config_space=config_space)
    history_container.load_history_from_json(file_out)

    return history_container






class HistoryContainer(object):
    def __init__(self, task_id, config_space=None):
        self.task_id = task_id
        self.num_objs = 1
        self.config_space = config_space  # for show_importance
        self.config_space_all = config_space

        self.info = None
        self.config_counter = 0
        self.data = collections.OrderedDict()  # only successful data
        self.data_all = collections.OrderedDict()
        self.incumbent_value = MAXINT
        self.incumbents = list()
        self.configurations = list()  # all configurations (include successful and failed)
        self.configurations_all = list()
        self.perfs = list()  # all perfs
        self.constraint_perfs = list()  # all constraints
        self.trial_states = list()  # all trial states
        self.elapsed_times = list()  # all elapsed times
        self.iter_times = list()
        self.external_metrics = list() # all external metrics
        self.internal_metrics = list() # all internal metrics
        self.resource = list() # all resource information
        self.contexts = list()

        self.update_times = list()  # record all update times

        self.successful_perfs = list()  # perfs of successful trials
        self.failed_index = list()
        self.transform_perf_index = list()

        self.global_start_time = time.time()
        self.scale_perc = 5
        self.perc = None
        self.min_y = None
        self.max_y = MAXINT

    def fill_default_value(self, config):
        values = {}
        for key in self.config_space_all._hyperparameters:
            if key in config.keys():
                values[key] = config[key]
            else:
                values[key] = self.config_space_all._hyperparameters[key].default_value

        c_new = Configuration(self.config_space_all, values)

        return c_new

    def update_observation(self, observation: Observation):
        self.update_times.append(time.time() - self.global_start_time)
        config = observation.config
        objs = observation.objs
        constraints = observation.constraints
        trial_state = observation.trial_state
        elapsed_time = observation.elapsed_time
        iter_time = observation.iter_time
        internal_metrics = observation.IM
        external_metrics = observation.EM
        resource = observation.resource
        info = observation.info
        context = observation.context

        if not self.info:
            self.info = info

        assert self.info == info

        self.configurations.append(config)
        self.configurations_all.append((self.fill_default_value(config)))
        if self.num_objs == 1:
            self.perfs.append(objs[0])
        else:
            self.perfs.append(objs)
        self.trial_states.append(trial_state)
        self.constraint_perfs.append(constraints)  # None if no constraint
        self.elapsed_times.append(elapsed_time)
        self.iter_times.append(iter_time)
        self.internal_metrics.append(internal_metrics)
        self.external_metrics.append(external_metrics)
        self.resource.append(resource)
        self.contexts.append(context)

        transform_perf = False
        failed = False
        if trial_state == SUCCESS and all(perf < MAXINT for perf in objs):
                # If infeasible, transform perf to the largest found objective value
            feasible = True

            if self.num_objs == 1:
                self.successful_perfs.append(objs[0])
            else:
                self.successful_perfs.append(objs)
            if feasible:
                if self.num_objs == 1:
                    self.add(config, objs[0])
                else:
                    self.add(config, objs)
            else:
                self.add(config, MAXINT)

            self.perc = np.percentile(self.successful_perfs, self.scale_perc, axis=0)
            self.min_y = np.min(self.successful_perfs, axis=0).tolist()
            self.max_y = np.max(self.successful_perfs, axis=0).tolist()

        else:
            # failed trial
            failed = True
            transform_perf = True

        cur_idx = len(self.perfs) - 1
        if transform_perf:
            self.transform_perf_index.append(cur_idx)
        if failed:
            self.failed_index.append(cur_idx)

    def get_contexts(self):
        return np.vstack(self.contexts)


    def add(self, config: Configuration, perf):
        if config in self.data:
            logger.warning('Repeated configuration detected!')
            return

        self.data[config] = perf
        self.data_all[self.fill_default_value(config)] = perf

        if len(self.incumbents) > 0:
            if perf < self.incumbent_value:
                self.incumbents.clear()
            if perf <= self.incumbent_value:
                self.incumbents.append((config, perf))
                self.incumbent_value = perf
        else:
            self.incumbent_value = perf
            self.incumbents.append((config, perf))

    def get_transformed_perfs(self, transform=None):
        # set perf of failed trials to current max
        transformed_perfs = self.perfs.copy()
        for i in self.transform_perf_index:
            transformed_perfs[i] = self.max_y

        transformed_perfs = np.array(transformed_perfs, dtype=np.float64)
        transformed_perfs = get_transform_function(transform)(transformed_perfs)
        return transformed_perfs

    

    def get_internal_metrics(self):
        return self.internal_metrics

    def get_perf(self, config: Configuration):
        return self.data[config]

    def get_all_perfs(self):
        return list(self.data.values())

    def get_all_configs(self):
        return list(self.data.keys())

    def empty(self):
        return self.config_counter == 0

    def get_incumbents(self):
        return self.incumbents

    def save_json(self, fn: str = "history_container.json"):
        data = []
        for i in range(len(self.perfs)):
            tmp = {
                'configuration': self.configurations_all[i].get_dictionary(),
                'external_metrics': self.external_metrics[i],
                'internal_metrics': self.internal_metrics[i],
                'resource': self.resource[i],
                'context': self.contexts[i],
                'trial_state': self.trial_states[i],
                'elapsed_time': self.elapsed_times[i],
                'iter_time': self.iter_times[i]
            }
            data.append(tmp)

        with open(fn, "w") as fp:
            json.dump({"info": self.info,  "data": data}, fp, indent=2)

    def transfer_configs(self, configs_dict):
        new_configs_dict = configs_dict
        for knob, value in configs_dict.items():
            param = self.config_space.get_hyperparameter(knob)
            if isinstance(param, CategoricalHyperparameter):
                param_range = param.choices
                if value in param_range:
                    continue
                for item in param_range:
                    if value.lower() == item.lower():
                        new_configs_dict[knob] = item
                        break
        return new_configs_dict  

    def load_history_from_json(self, fn: str = "history_container.json", load_num=None):  # todo: all configs
        try:
            with open(fn) as fp:
                all_data = json.load(fp)
        except Exception as e:
            logger.warning(
                'Encountered exception %s while reading runhistory from %s. '
                'Not adding any runs!', e, fn,
            )
            return

        info = all_data["info"]
        data = all_data["data"]

        y_variables = info['objs']
        c_variables = info['constraints']
        self.info = info

        assert len(self.info['objs']) == self.num_objs
        knobs_target = self.config_space.get_hyperparameter_names()
        knobs_default = self.config_space.get_default_configuration().get_dictionary()

        if not load_num is None:
            data = data[:load_num]
        for tmp in data:
            config_dict = tmp['configuration'].copy()
            knobs_source = tmp['configuration'].keys()
            knobs_delete = [knob for knob in knobs_source if knob not in knobs_target]
            knobs_add = [knob for knob in knobs_target if knob not in knobs_source]

            for knob in knobs_delete:
                config_dict.pop(knob)
            for knob in knobs_add:
                config_dict[knob] = knobs_default[knob]

            config_dict = self.transfer_configs(config_dict)
            config = Configuration(self.config_space, config_dict)
            em = tmp['external_metrics']
            im = tmp['internal_metrics']
            resource = tmp['resource']
            trial_state = tmp['trial_state']
            elapsed_time = tmp['elapsed_time']
            iter_time = tmp['iter_time'] if 'iter_time' in tmp.keys() else tmp['elapsed_time']
            context = tmp['context'] if 'context' in tmp.keys() else None
            res = dict(em, **resource)

            self.configurations.append(config)
            self.configurations_all.append(self.fill_default_value(config))
            self.trial_states.append(trial_state)
            self.elapsed_times.append(elapsed_time)
            self.iter_times.append(iter_time)
            self.internal_metrics.append(im)
            self.external_metrics.append(em)
            self.resource.append(resource)
            self.contexts.append(context)

            objs = self.get_objs(res, y_variables)
            if self.num_objs == 1:
                self.perfs.append(objs[0])
            else:
                self.perfs.append(objs)

            constraints = self.get_constraints(res, c_variables)
            self.constraint_perfs.append(constraints)

            transform_perf = False
            failed = False
            if trial_state == SUCCESS and all(perf < MAXINT for perf in objs):
                # If infeasible, transform perf to the largest found objective value
                feasible = True

                if self.num_objs == 1:
                    self.successful_perfs.append(objs[0])
                else:
                    self.successful_perfs.append(objs)
                if feasible:
                    if self.num_objs == 1:
                        self.add(config, objs[0])
                    else:
                        self.add(config, objs)
                else:
                    self.add(config, MAXINT)

                self.perc = np.percentile(self.successful_perfs, self.scale_perc, axis=0)
                self.min_y = np.min(self.successful_perfs, axis=0).tolist()
                self.max_y = np.max(self.successful_perfs, axis=0).tolist()

            else:
                # failed trial
                failed = True
                transform_perf = True

            cur_idx = len(self.perfs) - 1
            if transform_perf:
                self.transform_perf_index.append(cur_idx)
            if failed:
                self.failed_index.append(cur_idx)

    def get_objs(self, res, y_variables):
        try:
            objs = []
            for y_variable in y_variables:
                key = y_variable.strip().strip('-')
                value = res[key]
                if not y_variable.strip()[0] == '-':
                    value = - value
                objs.append(value)
        except:
            objs = [MAXINT] * self.num_objs

        return objs

    def get_constraints(self, res, constraints):
        if len(constraints) == 0:
            return None

        try:
            locals().update(res)
            constraintL = []
            for constraint in constraints:
                value = eval(constraint)
                constraintL.append(value)
        except:
            constraintL = []

        return constraintL

    def alter_configuration_space(self, new_space: ConfigurationSpace):
        names = new_space.get_hyperparameter_names()
        all_default_config_dict = self.config_space_all.get_default_configuration().get_dictionary()

        configurations = []
        data = collections.OrderedDict()

        for i in range(len(self.configurations)):
            config = self.configurations_all[i]
            config_new = {}
            for name in names:
                if name in config.get_dictionary().keys():
                    config_new[name] = config[name]
                else:
                    config_new[name] = all_default_config_dict[name]

            c_new = Configuration(new_space, config_new)
            configurations.append(c_new)
            perf = self.perfs[i]
            data[c_new] = perf

        self.configurations = configurations
        self.data = data
        self.config_space = new_space


    def get_shap_importance(self, config_space=None, return_dir=False, config_bench=None):
        import shap
        from lightgbm import LGBMRegressor
        from terminaltables import AsciiTable

        if config_space is None:
            config_space = self.config_space
        if config_space is None:
            raise ValueError('Please provide config_space to show parameter importance!')

        if config_bench is None:
            X_bench = self.config_space.get_default_configuration().get_array().reshape(1,-1)
        else:
            X_bench = config_bench.get_array().reshape(1,-1)

        X = np.array([list(config.get_array()) for config in self.configurations])
        Y = -  np.array(self.get_transformed_perfs())

        # Fit a LightGBMRegressor with observations
        lgbr = LGBMRegressor()
        lgbr.fit(X, Y)
        explainer = shap.TreeExplainer(lgbr)
        X_selected = X[Y>=self.get_default_performance()]
        if X_selected.shape[0] == 0:
            X_selected = X[Y >= np.quantile(Y, 0.9)]

        shap_values = explainer.shap_values(X)
        #shap_value_default = explainer.shap_values(X_bench)[-1]
        delta = shap_values #- shap_value_default
        delta = np.where(delta > 0, delta, 0)
        feature_importance = np.average(delta, axis=0)

        keys = [hp.name for hp in config_space.get_hyperparameters()]
        importance_list = []
        for i, hp_name in enumerate(keys):
            importance_list.append([hp_name, feature_importance[i]])
        importance_list.sort(key=lambda x: x[1], reverse=True)

        importance_dir = dict()
        for item in importance_list:
            importance_dir[item[0]] = item[1]

        # return the dir 
        if return_dir:
            return importance_dir
        

        for item in importance_list:
            item[1] = '%.6f' % item[1]
        table_data = [["Parameters", "Importance"]] + importance_list
        importance_table = AsciiTable(table_data).table

        return importance_table



    def get_default_performance(self):
        default_array = self.config_space.get_default_configuration().get_array()
        default_list = list()
        for i,config in enumerate(self.configurations):
            if (config.get_array() == default_array).all():
                default_list.append(self.get_transformed_perfs()[i])

        if not len(default_list):
            return self.get_transformed_perfs()[0]
        else:
            return  sum(default_list)/len(default_list)

 
