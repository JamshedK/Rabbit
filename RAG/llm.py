from RAG.rag import RAG
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from util.logger_config import logger
from langchain_core.output_parsers import StrOutputParser
import re
import random
from ConfigSpace import Configuration, ConfigurationSpace
import time

class LLM(RAG):
    def __init__(self, dbms, benchmark, vector_store_path, context, objective, search_space):
        super().__init__(dbms, benchmark, vector_store_path)
        self.context = context
        self.objective = 'latency' if objective=='-lat' else 'throughput'
        self.search_space = search_space
        self.model = ChatOpenAI(model=self.model_name, temperature=0.1)
        self.system_content = f'''You will be helping me with the knob tuning task for {self.dbms.name} database. '''


    def gen_candidates_llm(self, candidate_nums, target):
        human = f'''
            Machine Infomation: The machine running the {self.dbms.name} dbms has a RAM of {self.ram_size} GB, a CPU of {self.cpu_cores} cores, and a {self.disk_size} GB {self.disk_type} drive.
            WORKLOAD: {self.worklod_info}
            DATA: {self.data_info}
            The following examples demonstrate {self.dbms.name} database running on the above machine, workload and the data. These examples involve adjusting various knobs configurations to observe changes in {self.objective} metrics:
        '''
        for item in self.context:
            config = item['configuration']
            cost = item['cost']
            human += f"Configurations: {json.dumps(dict(config))}"
            human += f"Performance: {str(abs(cost))}\n"
        config_dict = [
                {
                    "knobname": hp.name,
                    "choices": hp.choices
                } for hp in self.search_space.get_hyperparameters()
            ]
        human +=  f"The database knob space is: {json.dumps(config_dict)}." + "\n"
        human_target = f"make the throughput of the database is {str(abs(int(target)))}" if target<0 else f"make the latency of the database is {str(abs(int(target)))}"
        human += """Please recommend %d new and different configurations (not in the examples), due to the exploration and utilization of the next round of the BO process. 
        These new configurations should be diverse and make some changes in knobs in order to %s.
        Each knob must contained within the knob space, Your response must only contain the predicted configurations, in the ##Configuration Format##:
        """%(candidate_nums, human_target)

        logger.info(human)

        messages = [
            SystemMessage(content=self.system_content),
            HumanMessage(content=human),
        ]
        parser = StrOutputParser()
        # chain = self.model | parser
        retries = 0
        time.sleep(2)
        while retries < 5:
            try:
                # 假设 api_call 是你要调用的函数
                result_mess = self.model.invoke(messages)
                result = parser.invoke(result_mess)
                self.add_tokens(result_mess)
                break
            except Exception as e:
                retries += 1
                print(f"Error encountered: {e}. Retrying {retries}...")
                time.sleep(8)

        logger.info(result)
        pattern = re.compile(r'\{.*?\}')
        matches = pattern.findall(result)
        samples = []
        for match in matches:
            samples.append(eval(match))
        return samples

    def prediction(self, configuration: Configuration):
        human = f'''
            Machine Infomation: The machine running the {self.dbms.name} dbms has a RAM of {self.ram_size} GB, a CPU of {self.cpu_cores} cores, and a {self.disk_size} GB {self.disk_type} drive.
            WORKLOAD: {self.worklod_info}
            DATA: {self.data_info}
            The following examples demonstrate {self.dbms.name} database running on the above machine, workload and the data. These examples involve adjusting various knobs configurations to observe changes in {self.objective} metrics:
        '''
        context = self.context[:]
        random.shuffle(context)
        for item in context:
            config = item['configuration']
            cost = item['cost']
            human += f"Configurations: {json.dumps(dict(config))}"
            human += f"Performance: {str(abs(cost))}\n"
        human += """Please combine the above information to predict the performance value of the following configuration, while also providing the confidence of the prediction: %s. 
        Your response should only contain the performance and the confidence in the JSON format of "{"Performance": value, "Confidence": value}"
        """%(json.dumps(dict(configuration)))

        logger.info(human)
        messages = [
            SystemMessage(content=self.system_content),
            HumanMessage(content=human),
        ]
        parser = StrOutputParser()
        chain = self.model | parser

        retries = 0
        while retries < 5:
            try:
                # 假设 api_call 是你要调用的函数
                result_mess = self.model.invoke(messages)
                result = parser.invoke(result_mess)
                self.add_tokens(result_mess)
                break
            except Exception as e:
                retries += 1
                print(f"Error encountered: {e}. Retrying {retries}...")
                import time
                time.sleep(8)
        logger.info(result)
        pattern = re.compile(r'\{.*?\}')
        matches = pattern.findall(result)
        if len(matches) >=1:
            return eval(matches[0])
        else:
            return None
    



