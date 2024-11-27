from space_optimizer.init_space import InitSpace
from smac import HyperparameterOptimizationFacade, Scenario, initial_design
import os
from RAG.init_rag import InitRAG
import json
from ConfigSpace import Configuration
import re
import sys
from util.logger_config import logger
import time
class InitStage(InitSpace):

    def __init__(self, dbms, test, timeout,  target_knobs_path, skill_path, init_configs_perfs, seed):
        super().__init__(dbms, test, timeout, target_knobs_path, seed)
        self.skill_path = skill_path
        self.init_configs_perfs = init_configs_perfs
        self.data = [] 
        

    # skill_path is the init_configs_path
    def transfer_configs(self, knobs_data):
        result_data = {}
        for index, knob in enumerate(self.target_knobs):
            info = self.dbms.knob_info[knob]
            
            knob_type = info["vartype"]
            if knob not in knobs_data.keys():
                logger.warning(f"No such {knob} knob in init_configs")
                continue
            if knob_type == "enum":
                sequence = []
                init_values = knobs_data[knob]
                if not info["enumvals"][0].isalpha():
                    result_data[knob] = init_values
                    continue
                for value in init_values:
                    for item in info["enumvals"]:
                        if value.lower() == item.lower():
                            sequence.append(item)
                            break
                    else:
                        sequence.append(info["enumvals"][0])

                result_data[knob] = sequence
                continue
            if knob_type == "bool":
                init_values = knobs_data[knob]
                sequence = []
                for value in init_values:
                    if value == "1" or value.lower() =="on":
                        sequence.append("on")
                    elif value == "0" or value.lower() == "off":
                        sequence.append("off")
                    else:
                        sequence.append("on")
                result_data[knob] = sequence
                continue
            
            init_values = knobs_data[knob]
            boot_value = info["reset_val"]
            unit = info["unit"]
            min_value = info["min_val"]
            max_value = info["max_val"]

            if unit is not None:
                unit = self._transfer_unit(unit)
                init_values = [(self._transfer_unit(value) / unit) for value in init_values]
                
            try:
                init_values = [self._type_transfer(knob_type, value) for value in init_values]
                boot_value = self._type_transfer(knob_type, boot_value)
                min_value = self._type_transfer(knob_type, min_value)
                max_value = self._type_transfer(knob_type, max_value)
            except:
                def match_num(value):
                    pattern = r"(\d+)"
                    match = re.match(pattern, value)
                    if match:
                        return match.group(1)
                    else:
                        return ""
                pattern = r"(\d+)"
                init_values = [self._type_transfer(knob_type, re.match(pattern, value).group(1)) for value in init_values if re.match(pattern, value) is not None]
                min_value = self._type_transfer(knob_type, match_num(min_value))
                max_value = self._type_transfer(knob_type, match_num(max_value))
                boot_value = self._type_transfer(knob_type, match_num(boot_value))
                

            if boot_value > sys.maxsize / 10:
                boot_value = sys.maxsize / 10
            sequence = []
            min_value = min(min_value, boot_value)
            max_value = max(max_value, boot_value)

            if int(max_value) > sys.maxsize:
                min_value = int(int(min_value) / 1000)
                max_value = int(int(max_value) / 1000)

            for value in init_values:
                if value < max_value and value > min_value:
                    sequence.append(value)
                elif value > max_value:
                    sequence.append(max_value)
                else:
                    sequence.append(min_value)


            if knob_type == "integer":
                sequence = [int(value) for value in sequence]
  
            elif knob_type == "real":
                sequence = [float(value) for value in sequence]


            result_data[knob] = sequence
        return result_data
    
    def get_resource(self):
        # need to change
        import time
        import psutil
        resource = {}
        cpu_util = psutil.cpu_percent(interval=1)
        io_counters = psutil.disk_io_counters()
        read_io = io_counters.read_bytes / (1024 ** 3)  # 转换为GB
        write_io = io_counters.write_bytes / (1024 ** 3)  # 转换为GB
        virtual_mem = psutil.virtual_memory().percent
        physical_mem = psutil.virtual_memory().used / (1024 ** 3)
        dirty_pages = psutil.disk_io_counters().write_count / (psutil.disk_io_counters().read_count + psutil.disk_io_counters().write_count)

        resource  = {
            "cpu": cpu_util,
            "readIO": read_io,
            "writeIO": write_io,
            "IO": read_io + write_io,
            "virtualMem": virtual_mem,
            "physical": physical_mem,
            "dirty": dirty_pages,
            "hit": 0.99,  # TODO
            "data": 0 # TODO
        }
        return resource


    def optimize(self, vector_store_path, init_number):
        if os.path.exists(self.init_configs_perfs):
            logger.info("Already get the init_configs_perfs")
            return
        if not os.path.exists(self.skill_path): # init_configs
            init_rag = InitRAG(self.dbms, self.test, vector_store_path, self.target_knobs, self.skill_path)
            # init_rag.load_data(init_number)
            init_rag.load_data_chain(init_number)
        
        # get the configs from init_configs and then set_and_replay
        with open(self.skill_path, 'r') as json_file:
            knobs_data = json.load(json_file)

        knobs_data = self.transfer_configs(knobs_data)


        
        for i in range(init_number):
            configs_dict = {}
            for key,value_list in knobs_data.items():
                configs_dict[key] = value_list[i]

            configs = Configuration(self.search_space, configs_dict)
            start_time = time.time()
            tps, lat, _ = self.set_and_replay_tps_and_lat(configs, usecnf=False)
            end_time = time.time()
            self.data.append({"configuration":dict(configs), 
                        "external_metrics": {"tps": tps, "lat":lat}, 
                        "internal_metrics":[], 
                        "resource": self.get_resource(),
                        "trial_state": 0,
                        "elapsed_time": end_time - start_time
                        })
        metric = "-lat" if self.test =="tpch" else "tps"
                
        result = {
            "info":{
                "objs": [
                    metric
                ],
                "constraints": []
            },
            "data":self.data
        }
        
        with open(self.init_configs_perfs, "w") as f:
            json.dump(result, f, indent=2)

        logger.info("Get the init_configs_perfs")
