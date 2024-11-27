from RAG.rag import RAG
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import SystemMessage
from langgraph.checkpoint import MemorySaver
from langchain.tools.retriever import create_retriever_tool
from util.logger_config import logger
from RAG.grah_rag import search_local, search_global
import json
import re
import random
import time
from util.utils import transfer_unit, type_transfer
class SuggestRAG(RAG):
    def __init__(self, dbms, benchmark, vector_store_path, target_knobs_path, history_transfer_path, suggested_knobs_path, rag_method):
        super().__init__(dbms, benchmark, vector_store_path)
        self.target_knobs_path = target_knobs_path
        self.history_transfer_path = history_transfer_path
        self.suggested_knobs_path = suggested_knobs_path
        self.target_knobs = self.get_target_knobs()
        self.memory = MemorySaver()
        self.thread_id = 0
        self.agents = []
        self.config = self.reset()
        self.rag_method = rag_method
        

    def reset(self):
        self.thread_id +=1
        config = {"configurable": {"thread_id": "thread"+str(self.thread_id)}}
        return config

    def start(self):
        knobs_list = [self.target_knobs[i:i + 20] for i in range(0, len(self.target_knobs), 20)]
        results = []
        for knobs in knobs_list:
            suggest_agent = SuggestAgent(self.dbms, self.benchmark, self.vector_stor_path, knobs, self.memory, self.config)
            history_agent = HistoryAgent(self.dbms, self.benchmark, self.vector_stor_path, knobs, self.memory, self.config, self.history_transfer_path)
            retriever_agent = RetrieverAgent(self.dbms, self.benchmark, self.vector_stor_path, knobs, self.memory, self.config)
            n = 0
            while n<1:
                if self.rag_method == 'graphRAG':
                    retriever_agent.retrieve()
                elif self.rag_method == 'selfRAG':
                    retriever_agent.retrieve_self()
                elif self.rag_method == 'prompt':
                    retriever_agent.retrieve_prompt()
                time.sleep(4)
                suggest_agent.suggest()
                time.sleep(4)
                history_agent.history()
                n += 1
            suggest_agent.suggest()
            time.sleep(4)
            # suggest_agent.write_json(self.suggested_knobs_path)
            results.extend(suggest_agent.write_list())
        with open(self.suggested_knobs_path, "w") as f:
            data = {}
            data["hyperparameters"] = results
            json.dump(data,f, indent=2)

    def start_wo_his(self, extra_knobs, extra_knobs_configs_path):
        knobs_list = [extra_knobs[i:i + 10] for i in range(0, len(extra_knobs), 10)]
        results = []

        for knobs in knobs_list:
            suggest_agent = SuggestAgent(self.dbms, self.benchmark, self.vector_stor_path, knobs, self.memory, self.config)
            suggest_agent.suggest_knowledge(knobs)
            # suggest_agent.write_json(self.suggested_knobs_path)
            results.extend(suggest_agent.write_list())
        data = {}
        logger.info(results)
        for item in results:
            suggested_values = item["suggested_values"]
            knob = item["knob"]
            info = self.dbms.knob_info[knob]
            unit = info["unit"]
            knob_type = info["vartype"] 
            if knob_type == "enum" or knob_type == "bool":
                data[item["knob"]] = suggested_values
                continue
            if unit is not None:
                unit = transfer_unit(unit)
                suggested_values = [(transfer_unit(value) / unit) for value in suggested_values]
            suggested_values = [type_transfer(knob_type, value) for value in suggested_values]
            data[item["knob"]] = suggested_values
            
        with open(extra_knobs_configs_path, "w") as f:
            json.dump(data,f, indent=2)
        
        
    def start_one(self):
        
        ans = []

        suggest_agent = SuggestAgent(self.dbms, self.benchmark, self.vector_stor_path, self.target_knobs, self.memory, self.config)
        history_agent = HistoryAgent(self.dbms, self.benchmark, self.vector_stor_path, self.target_knobs, self.memory, self.config, self.history_transfer_path)
        for knob in self.target_knobs:
            config = self.reset()
            suggest_agent.change_config(config)
            history_agent.change_config(config)
            target_knob = knob
            n = 0
            while n<1:
                suggest_agent.suggest(target_knob)
                time.sleep(4)
                history_agent.history(target_knob)
                n += 1
            suggest_agent.suggest(target_knob)
            ans_one = suggest_agent.write_dict()
            ans.append(ans_one)
        with open(self.suggested_knobs_path, "w") as f:
            data = {}
            data["hyperparameters"] = ans
            json.dump(data,f, indent=2)
        
        
    
    def get_target_knobs(self):
        with open(self.target_knobs_path, "r") as file:
            lines = file.readlines()
        target_knobs = [line.strip() for line in lines]
        return target_knobs


class SuggestAgent(RAG):
    def __init__(self, dbms, benchmark, vector_store_path, target_knobs, memory, config):
        super().__init__(dbms, benchmark, vector_store_path)
        self.target_knobs = target_knobs
        self.model = ChatOpenAI(model=self.model_name, temperature=0)
        self.name = "Suggest Agent"
        self.prefix = f"{self.name}: "
        self.memory = memory
        self.config = config
        self.result = ''
    
    def change_config(self, config):
        self.config = config

    def write_dict(self):
        pattern = r"```json(.*?)```"
        match = re.findall(pattern, self.result, re.DOTALL)
        if match:
            try:
                result = json.loads(match[0].strip())
                result = result
                return result
            except Exception:
                raise ValueError(f"Failed to parse: {self.result}")
        else:
            try:
                result = json.loads(self.result.strip())
                if isinstance(result, dict):
                    return result
                else:
                    raise ValueError(f"Parsed result is not a dictionary: {result}")
            except Exception:
                raise ValueError(f"Failed to parse: {self.result}")
            
    def write_list(self):
        pattern = r"```json(.*?)```"
        # pattern = r"```json\s*(\{.*?\}|\[.*?\])\s*```"
        # logger.info(self.result)
        match = re.findall(pattern, self.result, re.DOTALL)[0]
        try:
            if match.strip()[0] != '[':
                results = json.loads('['+match.strip()+']')
            else: 
                results = json.loads(match.strip())
            return results

        except Exception:
            raise ValueError(f"Failed to parse: {self.result}")
            return []
            

    def write_json(self, path):
        pattern = r"```json(.*?)```"
        match = re.findall(pattern, self.result, re.DOTALL)[0]
        try:
            results = json.loads('['+match.strip()+']')
            results = results
            with open(path, "w") as f:
                data = {}
                data["hyperparameters"] = results
                json.dump(data,f, indent=2)

        except Exception:
            raise ValueError(f"Failed to parse: {self.result}")
        
    def get_system_message(self):
        s = """
        Suppose you are an experienced DBA, and you are required to tune some knob of %s.
        --------------------------
        Target Knobs : %s
        Hardware Information: The machine running the dbms has a RAM of %d GB, a CPU of %d cores, and a %d GB %s drive.
        Workload Information: The benchmark is %s
        Data: %s
        --------------------------------------------------------
        TASK DESCRIPTION:
        Given the target knob name along with the workload information and the hardware information, your job is to offer at least three values that may lead to the best performance of the system under the workload and meet the hardware resource constraints, and offer at least three probability values (compared to the default values, the likelihood that the suggested values can improve the database performance). The values you need to provide are 'suggested_values', 'probability', 'min_values', and 'max_values'. If the suggested values fall within a continuous interval, provide the 'min_value' and 'max_value' for that interval.
        The result values should be numerical, and if a unit is needed, you can only choose from [KB, MB, GB, ms, s, min]; other units are not permitted.
        You should follow the steps:
        1. You need to give at least three recommended values ("suggested_values") for the target knobs above and the "probability" (compared to the default values, the likelihood that the suggested values can improve the database performance), according to the workload, the hardware, and the information that you get about the target knobs.
        2. You also need to return the recommended range("min_value" and "max_value") considering the hardware information and the current workload
        3. You should fine-tune your answers based on the messages provided by the History Agent (if have) so that your proposed knob configurations will achieve better performance for the database. The history agent will give suggestions based on the optimal configuration in similar historical tasks.
        4. Return the answer in the following JSON RESULT TEMPLATE for every knob and use ',' connect the answer for knobs. Just return the json result without others.
        JSON RESULT TEMPLATE:
        {{
            "knob": null, // it should be the knob name of the current knob name
            "suggested_values": [], // these should be exact values with a unit if needed (allowable units: KB, MB, GB, ms, s, min). When the value is unitless, they should be kept. All values should be string type.
            "probability": [] // every value should between 0 and 1, every suggested value should have a value
            "min_value": null, // change it whith the knowledge about the minimum value
            "max_value": null // change it whith the knowledge about the maximum value, it should be larger than min_value, please consider the hardware information and keep the unit of knob configuration itself. Don't return N/A
        }},
        """%(self.dbms.name, ", ".join(self.target_knobs), self.ram_size, self.cpu_cores, self.disk_size, self.disk_type, self.worklod_info, self.data_info)
        return s

    def get_system_message_one(self, knob):
        s = """
        Suppose you are an experienced DBA, and you are required to tune knob of %s.
        --------------------------
        Target Knob : %s
        Hardware Information: The machine running the dbms has a RAM of %d GB, a CPU of %d cores, and a %d GB %s drive.
        Workload Information: The benchmark is %s
        Data: %s
        --------------------------------------------------------
        TASK DESCRIPTION:
        Given the target knob name along with the workload information and the hardware information, your job is to offer at least three values that may lead to the best performance of the system under the workload and meet the hardware resource constraints, and offer at least three probability values (compared to the default values, the likelihood that the suggested values can improve the database performance). The values you need to provide are 'suggested_values', 'probability', 'min_values', and 'max_values'. If the suggested values fall within a continuous interval, provide the 'min_value' and 'max_value' for that interval.
        The result values should be numerical, and if a unit is needed, you can only choose from [KB, MB, GB, ms, s, min]; other units are not permitted.
        You should follow the steps:
        1. You need to give at least three recommended values ("suggested_values") for the target knob above and the "probability" (compared to the default value, the likelihood that the suggested values can improve the database performance), according to the workload, the hardware, and the information that you get about the target knob.
        2. You also need to return the recommended range("min_value" and "max_value") considering the hardware information and the current workload
        3. You should fine-tune your answers based on the messages provided by the History Agent (if have) so that your proposed knob configurations will achieve better performance for the database. The "history agent will" give the suggestion based on the optimal configuration in similar historical tasks.
        4. Return the answer in the following JSON RESULT TEMPLATE. Just return the json result without others.
        JSON RESULT TEMPLATE:
        {{
            "knob": null, // it should be the knob name of the current knob name
            "suggested_values": [], // these should be exact values with a unit if needed (allowable units: KB, MB, GB, ms, s, min). When the value is unitless, they should be kept. All values should be string type.
            "probability": [] // every value should between 0 and 1, every suggested value should have a value
            "min_value": null, // change it whith the knowledge about the minimum value
            "max_value": null // change it whith the knowledge about the maximum value, it should be larger than min_value, please consider the hardware information and keep the unit of knob configuration itself. Don't return N/A
        }}
        """%(self.dbms.name, knob, self.ram_size, self.cpu_cores, self.disk_size, self.disk_type, self.worklod_info, self.data_info)
        return s
    

    
    def suggest(self, knob=None):
        if knob is None:
            self.system_message =  self.get_system_message()
        else:
            self.system_message = self.get_system_message_one(knob)
        # embeddings = OpenAIEmbeddings()
        # docsearch = Chroma(
        #     persist_directory=self.vector_stor_path,
        #     embedding_function=embeddings,
        # )
        # retriever = docsearch.as_retriever(search_kwargs={"k":1})

        # retriever_tool = create_retriever_tool(
        #     retriever, 
        #     "knob_search",
        #     "Search for information about the specific knob. For any questions about knobs, you can use this tool!"
        # )

        tools = []
        query = "Please give your answer. \n" + self.prefix
        
        app = create_react_agent(self.model,  tools, messages_modifier=self.system_message, checkpointer=self.memory)


        messages = app.invoke({"messages": [("user", query)]}, self.config,)
        
        logger.info(messages)
        logger.info(messages["messages"][-1].content)
        self.result += messages["messages"][-1].content
    
    def suggest_knowledge(self, knobs):
        self.system_message = """
        Suppose you are an experienced DBA, and you are required to tune knobs of %s.
        """%(self.dbms.name)

        query = """
        Target Knobs : %s
        Hardware Information: The machine running the dbms has a RAM of %d GB, a CPU of %d cores, and a %d GB %s drive.
        Workload Information: The benchmark is %s
        Data: %s
        --------------------------------------------------------
        TASK DESCRIPTION:
        Given the target knob name along with the workload information and the hardware information, your job is to offer at least three values that may lead to the best performance of the system under the workload and meet the hardware resource constraints.
        The result values should be numerical, and if a unit is needed, you can only choose from [KB, MB, GB, ms, s, min]; other units are not permitted.
        Return the answer in the following JSON RESULT TEMPLATE for every knob and use ',' connect the answer for knobs. Just return the json result without others.
        "knob" should be the knob name of the current knob name. "suggested_values" should be exact values with a unit if needed (allowable units: KB, MB, GB, ms, s, min). When the value is unitless, they should be kept. All values should be string type.
        JSON RESULT TEMPLATE:
        {{
            "knob": null, 
            "suggested_values": [], 
        }},
        """%(", ".join(knobs), self.ram_size, self.cpu_cores, self.disk_size, self.disk_type, self.worklod_info, self.data_info)

        embeddings = OpenAIEmbeddings()
        docsearch = Chroma(
            persist_directory=self.vector_stor_path,
            embedding_function=embeddings,
        )
        retriever = docsearch.as_retriever(search_kwargs={"k":2})

        retriever_tool = create_retriever_tool(
            retriever, 
            "knob_search",
            "Search for information about the specific knob."
        )

        tools = [retriever_tool]
        
        app = create_react_agent(self.model,  tools, messages_modifier=self.system_message, checkpointer=self.memory)


        messages = app.invoke({"messages": [("user", query)]}, self.config,)
        
        logger.info(messages)
        logger.info(messages["messages"][-1].content)
        self.result = messages["messages"][-1].content
    

class HistoryAgent(RAG):
    def __init__(self, dbms, benchmark, vector_store_path, target_knobs, memory, config, history_transfer_path):
        super().__init__(dbms, benchmark, vector_store_path)
        self.target_knobs = target_knobs
        self.memory = memory
        self.config = config
        self.model = ChatOpenAI(model=self.model_name, temperature=0)
        self.name = "History Agent"
        self.prefix = f"{self.name}:"
        self.history_transfer_path = history_transfer_path
    
    def change_config(self,config):
        self.config = config
    
    def get_system_message_one(self, knob):
        with open(self.history_transfer_path, "r") as f:
            all_data = json.load(f)
        
        configs = {}
        configs[knob] = {"suggested_values":set()}

        for data in all_data:
            configs_all = data["Configs"]
            perf = data["Perf"]
            configs[knob]["suggested_values"].add(configs_all[knob])
        
        for knob,value in configs.items():
            if isinstance(random.sample(value["suggested_values"], 1)[0],(int, float)):
                value["min_value"] = min(value["suggested_values"])
                value["max_value"] = max(value["suggested_values"])
                # suggested_values = configs.pop("suggested_values")
            value["suggested_values"] = list(value["suggested_values"])
        
        history_s = json.dumps(configs)
        logger.info(history_s)
        s = """
        Suppose you are an experienced DBA, and you are required to tune some knob of %s.
        --------------------------
        Target Knobs : %s
        Hardware Information: The machine running the dbms has a RAM of %d GB, a CPU of %d cores, and a %d GB %s drive.
        Workload Information: The benchmark is %s
        Data: %s
        --------------------------------------------------------
        Similar Historial data:
        %s
        TASK DESCRIPTION:
        The similar historical data refers to the range of the best configuration composed of the historial tasks which are similar to current task. The data maybe the good range for the current task. 
        You should combine the similar historical data and the output of the "Suggest agent", and provide modification suggestions to "Suggest agent" and resons. 

        You should setp by setp as follow:
        Step1. You should check whether the range formed by min_value and max_value from "Suggest Agent" falls within or is as close as possible to the range given by historical data. If not, please provide modification suggestions for min_value and max_value based on the historical data.
        Step2. You need to check whether the suggested value provided by suggest_value from "Suggest Agent" is within or close to the range given by historical data. If it is not close, please provide modification suggestions to delete this suggested value or reduce the probability.
        Step3. Based on the suggested values in the "similar historical data" and the current task information, please add new suggested values and the "probability" (compared to the default values, the likelihood that the suggested values can improve the database performance) to supplement the current suggest_values from the "Suggest Agent".
        
        """%(self.dbms.name, knob, self.ram_size, self.cpu_cores, self.disk_size, self.disk_type, self.worklod_info, self.data_info, history_s)
        return s

    def get_system_message(self):

        with open(self.history_transfer_path, "r") as f:
            all_data = json.load(f)

        configs ={}

        not_in_knobs = []
        for knob in self.target_knobs:
            configs[knob] = {"suggested_values":set()}
            for data in all_data:
                configs_all = data["Configs"]
                if knob in list(configs_all.keys()):
                    configs[knob]["suggested_values"].add(configs_all[knob])
                else:
                    not_in_knobs.append(knob)
                    break

        
        for knob,value in configs.items():
            if knob in not_in_knobs:
                info = self.dbms.knob_info[knob]
                value = configs[knob]
                if info["vartype"] in ["integer", "real"]:
                    value["min_value"] = info["min_val"]
                    value["max_value"] = info["max_val"]
                    # suggested_values = configs.pop("suggested_values")
                value["suggested_values"] = [info['reset_val']]
            else:
                if isinstance(random.sample(value["suggested_values"], 1)[0],(int, float)):
                    value["min_value"] = min(value["suggested_values"])
                    value["max_value"] = max(value["suggested_values"])
                    # suggested_values = configs.pop("suggested_values")
                value["suggested_values"] = list(value["suggested_values"])
            
        
        history_s = json.dumps(configs)
        logger.info(history_s)
        s = """
        Suppose you are an experienced DBA, and you are required to tune some knob of %s.
        --------------------------
        Target Knobs : %s
        Hardware Information: The machine running the dbms has a RAM of %d GB, a CPU of %d cores, and a %d GB %s drive.
        Workload Information: The benchmark is %s
        Data: %s
        --------------------------------------------------------
        Similar Historial data:
        %s
        TASK DESCRIPTION:
        The similar historical data refers to the range of the best configuration composed of the historial tasks which are similar to current task. The data maybe the good range for the current task. 
        You should combine the similar historical data and the output of the "Suggest agent", and provide modification suggestions to "Suggest agent" and resons based on the current task information. 
        You should setp by setp as follow:

        Step1. You should check whether the range formed by min_value and max_value from "Suggest Agent" is slightly larger or closer than the range given by historical data. If not, please provide modification suggestions for min_value and max_value based on the historical data.
        Step2. You need to check whether the suggested value provided by suggest_value from "Suggest Agent" is within or close to the range given by historical data. If it is not close, please provide modification suggestions to delete this suggested value or reduce the probability.
        Step3. Based on the suggested values in the "similar historical data" and the current task information, please add new suggested values and the "probability" (compared to the default values, the likelihood that the suggested values can improve the database performance) to supplement the current suggest_values from the "Suggest Agent".
        
        """%(self.dbms.name, ",".join(self.target_knobs), self.ram_size, self.cpu_cores, self.disk_size, self.disk_type, self.worklod_info, self.data_info, history_s)
        return s

    def history(self, knob=None):
        if knob is None:
            self.system_message =  self.get_system_message()
        else:
            self.system_message = self.get_system_message_one(knob)
    
        # self.search_type = 'global'

        # @tool
        # async def query_grahRAG(question: str) -> str:
        #     """Search for information about the specific knob"""
        #     if self.search_type == 'local':
        #         logger.info('Local Search')
        #         ans = search_local(question)
        #     else:
        #         logger.info('Global Search')
        #         ans = search_global(question)
        #     return ans
            

        # embeddings = OpenAIEmbeddings()
        # docsearch = Chroma(
        #     persist_directory=self.vector_stor_path,
        #     embedding_function=embeddings,
        # )
        # retriever = docsearch.as_retriever(search_kwargs={"k":1})

        # retriever_tool = create_retriever_tool(
        #     retriever, 
        #     "knob_search",
        #     "Search for information about the specific knob. For any questions about knobs, you can use this tool!"
        # )

        tools = []
        query = "Please give your modification suggestions and resons.\n " + self.prefix
        
        app = create_react_agent(self.model, tools, messages_modifier=self.system_message, checkpointer=self.memory)

        # messages = app.invoke({"messages": [("user", "what was the Suggest Agent suggestions?")]}, self.config,)
        # logger.info(messages["messages"][-1].content)

        messages = app.invoke({"messages": [("user", query)]}, self.config,)
        logger.info(messages)
        # {
        #     "input": query,
        #     "output": messages["messages"][-1].content,
        # }
        logger.info(messages["messages"][-1].content)


class RetrieverAgent(RAG):
    def __init__(self, dbms, benchmark, vector_store_path, target_knobs, memory, config):
        super().__init__(dbms, benchmark, vector_store_path)
        self.target_knobs = target_knobs
        self.memory = memory
        self.config = config
        self.model = ChatOpenAI(model=self.model_name, temperature=0)
        self.name = "Retriver Agent"
        self.prefix = f"{self.name}:"
    
    def retrieve(self):
        self.system_message="execute the function query to look for context."
        self.search_type = 'global'

        @tool
        def query_grahRAG(question: str) -> str:
            """Query information from RAG search"""
            if self.search_type == 'local':
                logger.info('Local Search')
                ans = search_local(question)
            else:
                logger.info('Global Search')
                ans = search_global(question)
            
            return ans
        
        knob_list = [self.target_knobs[i:i + 5] for i in range(0, len(self.target_knobs), 5)]
        tools = [query_grahRAG]
        query = """Suppose you are an experienced DBA, and you are required to tune some knob of %s.
        --------------------------
        Target Knobs : %s
        Hardware Information: The machine running the dbms has a RAM of %d GB, a CPU of %d cores, and a %d GB %s drive.
        Workload Information The benchmark is %s
        Data: %s
        --------------------------------------------------------
        Task: 
        Please find the relevant information about the target knob configurations for the hardware and workload mentioned above. Each knob should have corresponding relevant details. 
        """%(self.dbms.name, ','.join(self.target_knobs), self.ram_size, self.cpu_cores, self.disk_size, self.disk_type, self.worklod_info, self.data_info) + self.prefix
        
        app = create_react_agent(self.model, tools, messages_modifier=self.system_message, checkpointer=self.memory)

        # messages = app.invoke({"messages": [("user", "what was the Suggest Agent suggestions?")]}, self.config,)
        # logger.info(messages["messages"][-1].content)

        messages = app.invoke({"messages": [("user", query)]}, self.config,)
        # logger.info(messages)
        # {
        #     "input": query,
        #     "output": messages["messages"][-1].content,
        # }
        logger.info(messages["messages"][-1].content)


    def retrieve_self(self):
        self.system_message="Only execute the function query to look for context."

        embeddings = OpenAIEmbeddings()
        docsearch = Chroma(
            persist_directory=self.vector_stor_path,
            embedding_function=embeddings,
        )
        retriever = docsearch.as_retriever(search_kwargs={"k": 5})

        query = create_retriever_tool(
            retriever,
            "knob_search",
            "Search for information about the specific knob. For any questions about knobs, you must use this tool!"
        )
        
        knob_list = [self.target_knobs[i:i + 5] for i in range(0, len(self.target_knobs), 5)]
        tools = [query]
        query = """Suppose you are an experienced DBA, and you are required to tune some knob of %s.
        --------------------------
        Target Knobs : %s
        Hardware Information: The machine running the dbms has a RAM of %d GB, a CPU of %d cores, and a %d GB %s drive.
        Workload Information The benchmark is %s
        Data: %s
        --------------------------------------------------------
        Task: 
        Please find the relevant information about the target knob configurations for the hardware and workload mentioned above. Each knob should have corresponding relevant details. 
        The target knobs consist of multiple lists, and each query is directed at one specific list within the target knobs
        """%(self.dbms.name, str(knob_list), self.ram_size, self.cpu_cores, self.disk_size, self.disk_type, self.worklod_info, self.data_info) + self.prefix
        
        app = create_react_agent(self.model, tools, messages_modifier=self.system_message, checkpointer=self.memory)

        # messages = app.invoke({"messages": [("user", "what was the Suggest Agent suggestions?")]}, self.config,)
        # logger.info(messages["messages"][-1].content)

        messages = app.invoke({"messages": [("user", query)]}, self.config,)
        # logger.info(messages)
        # {
        #     "input": query,
        #     "output": messages["messages"][-1].content,
        # }
        logger.info(messages["messages"][-1].content)

    def retrieve_prompt(self):
        self.system_message="Suppose you are an experienced DBA, and you are required to tune some knob of %s"%(self.dbms.name)

        
        knob_list = [self.target_knobs[i:i + 5] for i in range(0, len(self.target_knobs), 5)]
        tools = []
        query = """
        Target Knobs : %s
        Hardware Information: The machine running the dbms has a RAM of %d GB, a CPU of %d cores, and a %d GB %s drive.
        Workload Information The benchmark is %s
        Data: %s
        --------------------------------------------------------
        Task: 
        Please find the relevant information about the target knob configurations for the hardware and workload mentioned above. Each knob should have corresponding relevant details. 
        The target knobs consist of multiple lists, and each query is directed at one specific list within the target knobs
        """%(str(knob_list), self.ram_size, self.cpu_cores, self.disk_size, self.disk_type, self.worklod_info, self.data_info) + self.prefix
        
        app = create_react_agent(self.model, tools, messages_modifier=self.system_message, checkpointer=self.memory)

        # messages = app.invoke({"messages": [("user", "what was the Suggest Agent suggestions?")]}, self.config,)
        # logger.info(messages["messages"][-1].content)

        messages = app.invoke({"messages": [("user", query)]}, self.config,)
        # logger.info(messages)
        # {
        #     "input": query,
        #     "output": messages["messages"][-1].content,
        # }
        logger.info(messages["messages"][-1].content)

