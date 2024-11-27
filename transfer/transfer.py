from ConfigSpace import (
    UniformIntegerHyperparameter,
    UniformFloatHyperparameter,
    CategoricalHyperparameter,
    Configuration,
    ConfigurationSpace
)
import sys
import json
from transfer.history_container import HistoryContainer
import os
from util.logger_config import logger
from util.utils import check_random_state
from util.constants import MAXINT 
import numpy as np
from transfer.rgpe import RGPE

class Transfer:
    def __init__(self, task_id, data_repo_path, dbms, candidate_knobs_path, performance_metric, init_configs_perfs_path, test, extra_knobs_configs_path) -> None:
        self.task_id = task_id
        self.dbms = dbms
        self.test = test
        self.extra_knobs_configs_path = extra_knobs_configs_path
        self.hc_path = data_repo_path
        self.candidate_knobs_path = candidate_knobs_path
        self.init_configs_perfs_path = init_configs_perfs_path
        self.history_bo_data = list()
        self.method = "SMAC"
        self.surrogate_type = 'lightgbm'
        self.objs = performance_metric
        self.num_objs = len(self.objs)
        self.config_space = self.setup_configuration_space()
        self.history_container = HistoryContainer(task_id=self.task_id,
                                                  config_space=self.config_space)
        self.setup_transfer()

    def get_default_space(self, knob_name, info):
        boot_value = info["reset_val"]
        min_value = info["min_val"]
        max_value = info["max_val"]
        knob_type = info["vartype"]
        if knob_type == "integer":
            boot_value = int(boot_value)
            min_value = int(min_value)
            max_value = int(max_value)
            if boot_value >= sys.maxsize:
                boot_value = sys.maxsize / 10
            if max_value >= sys.maxsize:
                knob = UniformIntegerHyperparameter(
                    knob_name, 
                    min_value, 
                    sys.maxsize / 10,
                    default_value = boot_value
                )
            else:
                knob = UniformIntegerHyperparameter(
                    knob_name,
                    min_value,
                    max_value,
                    default_value = boot_value,
                )
        elif knob_type == "real":
            knob = UniformFloatHyperparameter(
                knob_name,
                float(min_value),
                float(max_value),
                default_value = float(boot_value)
            )
        elif knob_type == "enum":
            knob = CategoricalHyperparameter(
                knob_name,
                [str(enum_val) for enum_val in info["enumvals"]],
                default_value = str(boot_value),
            )
        elif knob_type == "bool":
            knob = CategoricalHyperparameter(
                knob_name,
                ["on", "off"],
                default_value = str(boot_value)
            )
        return knob

    def setup_configuration_space(self):

        # candidate knobs
        with open(self.candidate_knobs_path, 'r') as file:
            lines = file.readlines()
        self.candidate_knobs = [line.strip() for line in lines]
        
        # Load knob system config
        config_space = ConfigurationSpace()
        delete_knob = []
        for knob in self.candidate_knobs:
            # print(knob)
            info = self.dbms.knob_info[knob]
            if info is None or info.get("vartype") is None:
                delete_knob.append(knob)# this knob is not by the DBMS under specific version
                continue
            knob = self.get_default_space(knob, info)
            config_space.add_hyperparameter(knob)
        for knob in delete_knob:
            self.candidate_knobs.remove(knob)

        return config_space
    
 

    def get_history(self):
        return self.history_container
    
    def get_incumbent(self):
        return self.history_container.get_incumbents()
    
    def save_history(self):
        dir_path = os.path.join(self.hc_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        file_name = 'history_%s.json' % self.task_id
        return self.history_container.save_json(os.path.join(dir_path, file_name))
    
    def load_current_history(self):
        # TODO: check info
        fn = os.path.join(self.hc_path, 'history_%s.json' % self.task_id)
        if not os.path.exists(fn):
            # from init_configs_pers load data
            
            self.history_container.load_history_from_json(self.init_configs_perfs_path)
            logger.info('Start new RAGTuner task from init_configs_perfs')
        else:
            self.history_container.load_history_from_json(fn)
            logger.info('Load from {}'.format(fn))
    
    def load_history(self):
        files = os.listdir(self.hc_path)
        # config_space = self.setup_configuration_space(self.knob_config_file)
        for f in files:
            try:
                task_id = f.split('.')[0]
                fn = os.path.join(self.hc_path, f)
                history_container = HistoryContainer(task_id, config_space=self.config_space)
                history_container.load_history_from_json(fn)
                self.history_bo_data.append(history_container)
                logger.info("load history finished for {}".format(f))
            except Exception as e:
                logger.info('load history failed for {}, {}'.format(f,e))
    
    

    def get_similary_task(self,):
        if not hasattr(self, 'rgpe'):
            rng = check_random_state(100)
            seed = rng.randint(MAXINT)
            self.rgpe = RGPE(self.config_space, self.history_bo_data, seed, surrogate_type=self.surrogate_type, num_src_hpo_trial=-1, only_source=False)
        
        rank_loss_list = self.rgpe.get_ranking_loss(self.history_container)
        similarity_list = [1 - i for i in rank_loss_list]
        logger.info(similarity_list)
        # 0.63,0.5
        similarity_threhold = 0.60
        candidate_list, weight = list(), list()
        surrogate_list = self.history_bo_data + [self.history_container]
        for i in range(len(surrogate_list)):
            if similarity_list[i] > similarity_threhold:
                candidate_list.append(i)
                weight.append(similarity_list[i] )

        if not len(surrogate_list) -1 in candidate_list:
            candidate_list.append(len(surrogate_list) -1)
            weight.append(1)
        logger.info("Transfer result: \n")
        logger.info(weight)
        logger.info(candidate_list)
        
        # the similar surrogate_list
        sample_list = []
        for i in candidate_list:
            sample_list.append(surrogate_list[i])

        self.similar_tasks = sample_list
        self.weight = weight
        return sample_list, weight
    
    def get_incumbents_transfer(self, path):
        # best configs from similar tasks
        all_incumbents = []
        for task in self.similar_tasks:
            incumbents = task.get_incumbents()
            for configs, perf in incumbents:
                all_incumbents.append({"Configs":dict(configs), "Perf":perf, "Task":task.task_id})
        
        with open(path, "w") as f:
            json.dump(all_incumbents, f, indent=2)


        p_data = {}
        for d in all_incumbents:
            configs = d["Configs"]
            for knob, value in configs.items():
                if knob not in list(p_data.keys()):
                    p_data[knob] = set()
                p_data[knob].add(value)

        for knob, value in p_data.items():
            p_data[knob] = list(value)

        with open(self.extra_knobs_configs_path, "w") as f:
            json.dump(p_data, f,indent=4)
        
        




    def setup_transfer(self):
        if len(self.history_bo_data)==0 :
            self.load_history()
        # load current history
        self.load_current_history()

        