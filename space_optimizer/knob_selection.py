import os
import json
from util.logger_config import logger
from RAG.select_rag import SelectRAG
class KnobSelection:
    def __init__(self, dbms, benchmark, vector_store_path, version, candidate_knobs_path, target_knobs_path, similar_task, weights, key_knobs_num, append_ratio, rag_method) -> None:
        self.dbms = dbms
        self.benchmark = benchmark
        self.vector_store_path = vector_store_path
        self.version = version
        self.candidate_knobs_path = candidate_knobs_path
        self.candidate_knobs = self.get_candidate_knobs()
        self.target_knobs_path = target_knobs_path
        self.similar_task = similar_task
        self.weights = weights
        self.key_knobs_num = key_knobs_num
        self.append_ratio = append_ratio
        self.rag_method = rag_method
        if os.path.exists(self.target_knobs_path):
            logger.info(f"Knobs already selected for {self.dbms.name}")
        else:
            if self.similar_task is not None:
                self.target_knobs = self.get_target_knobs()
            else:
                self.target_knobs = self.get_target_knobs_wo_his()

    

    def get_candidate_knobs(self):

        with open(self.candidate_knobs_path, "r") as f:
            lines = f.readlines()
        candidate_knobs = [line.strip() for line in lines]
        return candidate_knobs

    
    def get_target_knobs_wo_his(self):
        # 首先使用文档辅助的 llm 生成重要的 key knobs；然后进行旋钮的扩增
        select_rag = SelectRAG(self.dbms, self.benchmark, self.vector_store_path, self.version, None, self.candidate_knobs)
        sorted_knobs = select_rag.load_data()
        base, ext = os.path.splitext(self.target_knobs_path)
        self.key_knobs_fine = f"{base}_all_sorted{ext}"
        with open(self.key_knobs_fine, 'w') as f:
            for knob in sorted_knobs:
                f.write(knob+"\n")

        target_knobs_base = sorted_knobs[0:self.key_knobs_num]
        target_knobs = self.target_knobs_rag(target_knobs_base)
        with open(self.target_knobs_path, "w") as f:
            for knob in target_knobs:
                f.write(knob+"\n")
        return target_knobs

    def get_target_knobs(self):
        target_knobs_base = self.target_knobs_similarity()
        
        target_knobs = self.target_knobs_rag(target_knobs_base)
        with open(self.target_knobs_path, "w") as f:
            for knob in target_knobs:
                f.write(knob+"\n")
        return target_knobs


    
    def target_knobs_rag(self, target_knobs):
        select_rag = SelectRAG(self.dbms, self.benchmark, self.vector_store_path, self.version, target_knobs)
        knobs_num = self.key_knobs_num // self.append_ratio
        if self.rag_method == "graphRAG":
            final_knobs = select_rag.load_data_graph(knobs_num)
        elif self.rag_method == 'selfRAG':
            final_knobs = select_rag.load_data_self(knobs_num)
        elif self.rag_method == 'prompt':
            final_knobs = select_rag.load_data_self(knobs_num)
            
        logger.info(final_knobs)
        return final_knobs

    def target_knobs_similarity(self):
         # TF-IDF
        final_scores = {}
        final_nums = {}
        for i in range(len(self.similar_task)):
            important_knobs_shap = self.similar_task[i].get_shap_importance(return_dir=True)
            logger.info(important_knobs_shap)

            sorted_knobs = sorted(important_knobs_shap.items(), key=lambda item: item[1], reverse=True)
            weight = self.weights[i]

            for pos, (knob, shap_value) in enumerate(sorted_knobs, start=1):
                if shap_value == 0: 
                    score = 0
                    break
                else:
                    # score = 1/(n+1-pos)
                    score = 1/ pos
                    if knob in final_scores:
                        final_scores[knob] += score * weight
                    else:
                        final_scores[knob] = score * weight
 
                
                

        final_sorted_scores = dict(sorted(final_scores.items(), key=lambda x: x[1], reverse=True))
        logger.info(final_sorted_scores)
        # test, need to change
        base, ext = os.path.splitext(self.target_knobs_path)
        self.key_knobs_fine = f"{base}_all_sorted{ext}"
        with open(self.key_knobs_fine, 'w') as f:
            for knob in list(final_sorted_scores.keys()):
                f.write(knob+"\n")

        return list(final_sorted_scores.keys())[0:self.key_knobs_num]



        



