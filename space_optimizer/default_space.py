from abc import abstractmethod
from config_recommender.workload_runner import BenchbaseRunner
import time
import sys
import threading
import re
import time
from util.logger_config import logger
from ConfigSpace import (
    ConfigurationSpace,
    UniformIntegerHyperparameter,
    UniformFloatHyperparameter,
    CategoricalHyperparameter,
)


class DefaultSpace:
    """ Base template of RAGTuner"""
    def __init__(self, dbms, test, timeout, target_knobs_path, seed=1):
        self.dbms = dbms
        self.seed = seed if seed is not None else 1
        self.test = test
        self.timeout = timeout
        self.target_knobs_path = target_knobs_path
        self.round = 0
        self.summary_path = "./optimization_results/temp_results"
        self.benchmark_copy_db = ['tpcc', 'twitter', "sibench", "voter", "tatp", "smallbank", "seats", "ycsb"]   # Some benchmark will insert or delete data, Need to be rewrite each time.
        self.benchmark_latency = ['tpch', 'wikipedia']
        self.search_space = ConfigurationSpace()
        self.target_knobs = self.knob_select()
        self.target_knobs_num = len(self.target_knobs)
        if self.test in self.benchmark_copy_db:
            # self.dbms.create_template(self.test)
            self.penalty = self.get_default_result()
        else:
            self.penalty = 41383858
        logger.info(f"DEFAULT : {self.penalty}")
        self.log_file = f"./optimization_results/{self.dbms.name}/log/{self.seed}_log.txt"
        self.init_log_file()
        self.prev_end = 0


    def init_log_file(self):
        with open(self.log_file, 'w') as file:
            file.write(f"Round\tStart\tEnd\tBenchmark_Elapsed\tTuning_overhead\n")

    def _log(self, begin_time, end_time):
        if self.round == 1:
            self.prev_end = begin_time
        with open(self.log_file, 'a') as file:
            file.write(f"{self.round}\t{begin_time}\t{end_time}\t{end_time-begin_time}\t{begin_time-self.prev_end}\n")
        self.prev_end = end_time

    def _transfer_unit(self, value):
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
    
    def _type_transfer(self, knob_type, value):
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
        
    def _reload_data(self):
        print("Reloading the data")
        self.dbms._disconnect()
        self.dbms._connect(f"{self.test}_template")
        self.dbms.copy_db(target_db="benchbase", source_db=f"{self.test}_template")
        print("Reloading completed")
        time.sleep(6)
        self.dbms._disconnect()
        time.sleep(4)
        self.dbms._connect('benchbase')
        time.sleep(6)

    def knob_select(self):
        """ 
            Select which knobs to be tuned, store the names in 'self.target_knobs' 
            Default implementation is to use fixed knobs. Provide the path to the file containing the knobs' names.
        """
        target_knobs = []
        with open(self.target_knobs_path, 'r') as file:
            lines = file.readlines()
        for line in lines:
            if not self.dbms.knob_info[l:=line.strip()]["vartype"] == "string":
                target_knobs.append(l)
        logger.info(f"Target knobs: {target_knobs}")
        return target_knobs
    
    def get_default_space(self, knob_name, info):
        boot_value = info["reset_val"]
        min_value = info["min_val"]
        max_value = info["max_val"]
        knob_type = info["vartype"]
        knob = None
        if knob_type == "integer":
            if int(boot_value) >= sys.maxsize:
                boot_value = sys.maxsize / 10
            if int(max_value) >= sys.maxsize:
                knob = UniformIntegerHyperparameter(
                    knob_name, 
                    int(min_value), 
                    int(sys.maxsize / 10),
                    default_value = int(boot_value)
                )
            else:
                knob = UniformIntegerHyperparameter(
                    knob_name,
                    int(min_value),
                    int(max_value),
                    default_value = int(boot_value),
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

    def get_default_result(self):
        logger.info("Test the result in default conf")
        dbms = self.dbms
        print(f"--- Restore the dbms to default configuration ---")
        dbms.reset_config()
        dbms.reconfigure()

        try:
            # if self.test in self.benchmark_copy_db:
            # # reload the data
            #     self._reload_data()
                
            logger.info("Begin to run benchbase...")
            runner = BenchbaseRunner(dbms=dbms, test=self.test, target_path=self.summary_path)
            runner.clear_summary_dir()
            t = threading.Thread(target=runner.run_benchmark)
            t.start()
            t.join()
            throughput, average_latency = runner.get_throughput(), runner.get_latency()
        except Exception as e:
            logger.info(f'Exception for {self.test}: {e}')

        if self.test not in self.benchmark_latency:
            return throughput
        else:
            return average_latency


    def set_and_replay(self, config, seed=0):
        begin_time = time.time()
        cost = self.set_and_replay_ori(config, seed, False)
        end_time = time.time()
        self._log(begin_time, end_time)
        return cost
    
    def set_and_replay_tps_and_lat(self, config, seed=0, usecnf=False):
        begin_time = time.time()
        self.round += 1
        logger.info(f"Tuning round {self.round} ...")
        dbms = self.dbms
        logger.info(f"--- Restore the dbms to default configuration ---")
        dbms.reset_config()
        dbms.reconfigure()
        # reload the data
        # if self.test in self.benchmark_copy_db:
        #     self._reload_data()

        if usecnf:
            print(f"--- knob setting procedure ---")
            #这个函数目前仅针对mysql设计的。
            dbms.write_to_reset_knob_config_and_check("/etc/mysql/my.cnf", self.target_knobs, config)

        if not usecnf:
            print(f"--- knob setting procedure ---")
            for knob in self.target_knobs:
                value = config[knob]
                dbms.set_knob(knob, value)
            dbms.reconfigure()
        
        if dbms.failed_times == 4:
            print("连接失败，本轮惩罚增大！")
            if self.test not in self.benchmark_latency:
                tps = int(self.penalty) / 2
                lat = 1000000
            else:
                lat = self.penalty * 2
                tps = 0
            end_time = time.time()
            return (tps, lat, end_time - begin_time)
            

        try:
            logger.info("Begin to run benchbase...")
            runner = BenchbaseRunner(dbms=dbms, test=self.test, target_path=self.summary_path)
            runner.clear_summary_dir()
            t = threading.Thread(target=runner.run_benchmark)
            t.start()
            t.join(timeout=self.timeout)
            if t.is_alive():
                logger.info("Benchmark is still running. Terminate it now.")
                runner.process.terminate()
                time.sleep(2)
                raise RuntimeError("Benchmark is still running. Terminate it now.") 
            else:
                logger.info("Benchmark has finished.")
                if runner.check_sequence_in_file():  ### 如果query出错
                    raise RuntimeError("ERROR in Query.") 
                tps, lat = runner.get_throughput(), runner.get_latency()

                if self.test not in self.benchmark_latency and tps < self.penalty:
                    self.penalty = tps
                if self.test in self.benchmark_latency and lat > self.penalty:
                    self.penalty = lat
                

        except Exception as e:
            logger.info(f'Exception for {self.test}: {e}')
            # update worst_perf
            if self.test not in self.benchmark_latency:
                tps =  int(self.penalty) / 2
                lat = 1000000
                ###tpch
            else:
                lat = self.penalty * 2
                tps = 0
    
        end_time = time.time()
        self._log(begin_time, end_time)
        return (tps, lat, end_time - begin_time)


    def set_and_replay_ori(self, config, seed=0, usecnf=False):
        self.round += 1
        logger.info(f"Tuning round {self.round} ...")
        dbms = self.dbms

        print(f"--- Restore the dbms to default configuration ---")
        dbms.reset_config()
        dbms.reconfigure()

        # reload the data
        # if self.test in self.benchmark_copy_db:
        #     self._reload_data()
        if usecnf:
            print(f"--- knob setting procedure ---")
            #这个函数目前仅针对mysql设计的。
            dbms.write_to_reset_knob_config_and_check("/etc/mysql/my.cnf", self.target_knobs, config)

        if not usecnf:
            print(f"--- knob setting procedure ---")
            for knob in self.target_knobs:
                value = config[knob]
                dbms.set_knob(knob, value)
            dbms.reconfigure()

        if dbms.failed_times == 4:
            print("连接失败，本轮惩罚增大！")
            if self.test not in self.benchmark_latency:
                return -int(self.penalty) / 2
            else:
                return self.penalty * 2
            
        try:
            logger.info("Begin to run benchbase...")
            runner = BenchbaseRunner(dbms=dbms, test=self.test, target_path=self.summary_path)
            runner.clear_summary_dir()
            t = threading.Thread(target=runner.run_benchmark)
            t.start()
            t.join(timeout=self.timeout)
            if t.is_alive():
                logger.info("Benchmark is still running. Terminate it now.")
                runner.process.terminate()
                time.sleep(2)
                raise RuntimeError("Benchmark is still running. Terminate it now.") 
            else:
                logger.info("Benchmark has finished.")
                if runner.check_sequence_in_file():  ### 如果query出错
                    raise RuntimeError("ERROR in Query.") 
                throughput, average_latency = runner.get_throughput(), runner.get_latency()

                if self.test not in self.benchmark_latency and throughput < self.penalty:
                    self.penalty = throughput
                if self.test in self.benchmark_latency and average_latency > self.penalty:
                    self.penalty = average_latency

        except Exception as e:
            logger.info(f'Exception for {self.test}: {e}')
            # update worst_perf
            if self.test not in self.benchmark_latency:
                return -int(self.penalty) / 2
                ###tpch
            else:
                return self.penalty * 2
    

        if self.test not in self.benchmark_latency:
            return -throughput
        else:
            return average_latency


    @abstractmethod
    def define_search_space(self):
        pass