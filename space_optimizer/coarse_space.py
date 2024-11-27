from space_optimizer.default_space import DefaultSpace
from dbms.mysql import MysqlDBMS
import sys
import os
import json
import re
import numpy as np
from util.logger_config import logger
from smac import HyperparameterOptimizationFacade, Scenario, initial_design
from ConfigSpace import (
    CategoricalHyperparameter,
    Constant,
    EqualsCondition,
)

class CoarseSpace(DefaultSpace):

    def __init__(self, dbms, test, timeout, target_knobs_path, skill_path, incumbents_transfer_path, extra_knobs_configs_path, seed):
        super().__init__(dbms, test, timeout, target_knobs_path, seed)
        # self.factors = [0, 0.25, 0.5]
        self.skill_path = skill_path
        self.incumbents_transfer_path =  incumbents_transfer_path
        self.extra_knobs_configs_path = extra_knobs_configs_path
        self.define_search_space()


    def define_search_space(self):
        n_min = 2
        n_max = 5
        
        # from init_stage load the good start point
        with open(self.incumbents_transfer_path, "r") as f:
            transfer_incumnbent = json.load(f)[-1]
        transfer_configs = transfer_incumnbent["Configs"]

        with open(self.skill_path, 'r') as json_file:
            all_data = json.load(json_file)
        # print('skill_data: ',all_data["hyperparameters"])
        all_knobs_data = all_data["hyperparameters"]

        with open(self.extra_knobs_configs_path, "r") as f:
            extra_knobs_configs = json.load(f)

        knobs_data = {}
        for data in all_knobs_data:
            knobs_data[data["knob"]] = data
    
        for index, knob in enumerate(self.target_knobs):
            # print(knob)
            info = self.dbms.knob_info[knob]
            if info is None:
                # self.target_knobs.remove(knob) # this knob is not by the DBMS under specific version
                continue

            knob_type = info["vartype"] 
            if knob_type == "enum" or knob_type == "bool":
                knob = self.get_default_space(knob, info)
                self.search_space.add_hyperparameter(knob)
                continue
            
            if knob in list(knobs_data.keys()):
                data = knobs_data[knob]
            else:
                data["suggested_values"] = []
                data["probaility"] = []
                data["min_value"] = None
                data["max_value"] = None

            suggested_values = data["suggested_values"]
            C_list = data["probability"]
            
            if knob in list(transfer_configs.keys()):
                transfer_boot_value = transfer_configs[knob]
            else:
                transfer_boot_value = info["reset_val"]

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
                transfer_boot_value = self._type_transfer(knob_type, transfer_boot_value)
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
                transfer_boot_value = self._type_transfer(knob_type, match_num(transfer_boot_value))
                
            if boot_value > sys.maxsize / 10:
                boot_value = sys.maxsize / 10
            sequence = []
            min_value = min(min_value, boot_value)
            max_value = max(max_value, boot_value)

            if transfer_boot_value <= max_value and transfer_boot_value >= min_value:
                boot_value = transfer_boot_value

            if max_value > sys.maxsize / 10:
                max_value = sys.maxsize / 10
            
            if min_value > sys.maxsize / 10:
                min_value = sys.maxsize / 10
            
            if knob in list(extra_knobs_configs.keys()):
                extra_configs = extra_knobs_configs[knob]
                max_value = max(max_value, max(extra_configs))
                min_value = min(min_value, min(extra_configs))

            
            for index, value in enumerate(suggested_values):
                if value > max_value or value < min_value:
                    continue
                C = C_list[index]
                n = int(n_min * np.exp((np.log(n_max / n_min) * (1 - C))))
                logger.info("n: "+ str(n))
                for i in range(1,n+1):
                    sample = value + (i * (1 - C)  / n ) * (max_value - value) 
                    sequence.append(sample)

                    sample = value + (i * (1 - C)  / n ) * (min_value - value) 
                    sequence.append(sample)

                sequence.append(value)
                

            # if a suggested value is not given but a min_val or masx_val is suggested in skill library, equidistant sample.
            if sequence == [] and (not min_from_sys or not max_from_sys):
                for factor in [0.25, 0.5, 0.75]:
                    sequence.append(boot_value + factor * (max_value - boot_value)) 
                if not min_from_sys:
                    sequence.append(min_value)
                if not max_from_sys:
                    sequence.append(max_value)
            sequence.append(boot_value)
        
         

            if knob_type == "integer":  
                sequence = [int(value) for value in sequence]
                sequence = list(set(sequence))
                sequence.sort()
                normal_para = CategoricalHyperparameter(
                    knob,
                    [str(value) for value in sequence],
                    default_value = str(boot_value),
                )
          
                self.search_space.add_hyperparameter(normal_para)
                
            elif knob_type == "real":
                sequence = [float(value) for value in sequence]
                sequence = list(set(sequence))
                sequence.sort()

                normal_para = CategoricalHyperparameter(
                    knob,
                    [str(value) for value in sequence],
                    default_value = str(boot_value),
                )
               
                self.search_space.add_hyperparameter(normal_para)
            logger.info("sequence: "+str(sequence))
            
