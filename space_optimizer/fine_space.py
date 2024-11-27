from abc import ABC, abstractmethod
from space_optimizer.default_space import DefaultSpace
from dbms.mysql import MysqlDBMS
from dbms.postgres import PgDBMS
import sys
import os
import json
import re
from smac import HyperparameterOptimizationFacade, Scenario, initial_design, intensifier
from ConfigSpace import (
    UniformIntegerHyperparameter,
    UniformFloatHyperparameter,
    CategoricalHyperparameter,
    Constant,
    Configuration,
    EqualsCondition,
)
from util.logger_config import logger

class FineSpace(DefaultSpace):

    def __init__(self, dbms, test, timeout, target_knobs_path, skill_path, incumbents_transfer_path, extra_knobs_configs_path, seed):
        super().__init__(dbms, test, timeout, target_knobs_path, seed)
        self.skill_path = skill_path
        self.incumbents_transfer_path =  incumbents_transfer_path
        self.extra_knobs_configs_path = extra_knobs_configs_path
        self.define_search_space()
        self.coarse_path = f"./optimization_results/{self.dbms.name}/coarse/{self.seed}/runhistory.json"


    def define_search_space(self):
        with open(self.skill_path, 'r') as json_file:
            suggested_knob_data = json.load(json_file)["hyperparameters"]

        with open(self.extra_knobs_configs_path, "r") as f:
            extra_knobs_configs = json.load(f)
        
        knob_data = {data["knob"]:data for data in suggested_knob_data}
        
        suggested_knob_name = [knob["knob"] for knob in suggested_knob_data]
        
        for knob in self.target_knobs:
            info = self.dbms.knob_info[knob]
            if info is None:
                self.target_knobs.remove(knob) # this knob is not by the DBMS under specific version
                continue

            knob_type = info["vartype"] 
            if knob_type == "enum" or knob_type == "bool":
                knob = self.get_default_space(knob, info)
                self.search_space.add_hyperparameter(knob)
                continue
        
            if knob in suggested_knob_name:
                data = knob_data[knob]
                logger.info(f"Defining fine search space for knob: {knob}")
                suggested_values = data["suggested_values"]
                boot_value = info["reset_val"]
                unit = info["unit"]

                # hardware constraint if exists
                min_from_sys, max_from_sys = False, False
                min_value = data["min_value"]
                if min_value is None:
                    min_value = info["min_val"]
                    min_from_sys = True
                
                max_value = data["max_value"]
                if max_value is None:
                    max_value = info["max_val"]
                    max_from_sys = True

                if not min_from_sys:
                    if unit:
                        unit = self._transfer_unit(unit)
                        min_value = self._transfer_unit(min_value) / unit

                        min_value = self._type_transfer(knob_type, min_value)
                        sys_min_value = self._type_transfer(knob_type, info["min_val"])

                        if min_value < sys_min_value:
                            min_value = sys_min_value

                if not max_from_sys:
                    if unit:
                        unit = self._transfer_unit(unit)
                        max_value = self._transfer_unit(max_value) / unit

                        max_value = self._type_transfer(knob_type, max_value)
                        sys_max_value = self._type_transfer(knob_type, info["max_val"])
                        if max_value > sys_max_value:
                            max_value = sys_max_value
          
                # unit transformation
                if unit is not None:
                    unit = self._transfer_unit(unit)
                    suggested_values = [(self._transfer_unit(value) / unit) for value in suggested_values]
                
                # type transformation
                try:
                    suggested_values = [self._type_transfer(knob_type, value) for value in suggested_values]
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
                    suggested_values = [self._type_transfer(knob_type, re.match(pattern, value).group(1)) for value in suggested_values if re.match(pattern, value) is not None]
                    min_value = self._type_transfer(knob_type, match_num(min_value))
                    max_value = self._type_transfer(knob_type, match_num(max_value))
                    boot_value = self._type_transfer(knob_type, match_num(boot_value))
                    
                if boot_value > sys.maxsize / 10:
                    boot_value = sys.maxsize / 10

                # the search space of fine-grained stage should be superset of that of coarse stage
                coarse_sequence = []
                if boot_value > sys.maxsize / 10:
                    boot_value = sys.maxsize / 10


                min_value = min(min_value, boot_value)
                max_value = max(max_value, boot_value)
                # scale up and down the suggested value
                for value in suggested_values:
             
                    coarse_sequence.append(value)
                
                
                if coarse_sequence == [] and (not min_from_sys or not max_from_sys):
                    for factor in [0.25, 0.5, 0.75]:
                        coarse_sequence.append(boot_value + factor * (max_value - boot_value)) 
                    if not min_from_sys:
                        coarse_sequence.append(min_value)
                    if not max_from_sys:
                        coarse_sequence.append(max_value)
                coarse_sequence.append(boot_value)

                if max_value > sys.maxsize / 10:
                    max_value = sys.maxsize / 10
                
                if min_value > sys.maxsize / 10:
                    min_value = sys.maxsize / 10

                coarse_sequence = [value for value in coarse_sequence if value < sys.maxsize / 10]
                
                
                
                if knob_type == "integer":  
                    coarse_sequence = [int(value) for value in coarse_sequence]
                    min_value = min(min_value, min(coarse_sequence))
                    max_value = max(max_value, max(coarse_sequence))

                    if knob in list(extra_knobs_configs.keys()):
                        extra_configs = extra_knobs_configs[knob]
                        max_value = max(max_value, max(extra_configs))
                        min_value = min(min_value, min(extra_configs))
                        
                    normal_para = UniformIntegerHyperparameter(
                        knob, 
                        int(min_value), 
                        int(max_value),
                        default_value = int(boot_value),
                    )
                    self.search_space.add_hyperparameter(normal_para)
                    
                elif knob_type == "real":
                    coarse_sequence = [float(value) for value in coarse_sequence]
                    min_value = min(min_value, min(coarse_sequence))
                    max_value = max(max_value, max(coarse_sequence))

                    if knob in list(extra_knobs_configs.keys()):
                        extra_configs = extra_knobs_configs[knob]
                        max_value = max(max_value, max(extra_configs))
                        min_value = min(min_value, min(extra_configs))
                        
                    normal_para = UniformFloatHyperparameter(
                        knob,
                        float(min_value),
                        float(max_value),
                        default_value = float(boot_value),
                    )
                    self.search_space.add_hyperparameter(normal_para)
            else:
                info = self.dbms.knob_info[knob]
                if info is None:
                    continue
                knob = self.get_default_space(knob, info)
                self.search_space.add_hyperparameter(knob)

    

            