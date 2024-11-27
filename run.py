from configparser import ConfigParser
import argparse
import time
import os
from dotenv import load_dotenv
from dbms.postgres import PgDBMS
from dbms.mysql import  MysqlDBMS
from config_recommender.init_stage import InitStage
from transfer.transfer import Transfer
from util.logger_config import logger
from space_optimizer.knob_selection import KnobSelection
from RAG.suggest_graph_rag import SuggestRAG

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=str, default="mysql")
    parser.add_argument("--test", type=str, default="tpcc")
    parser.add_argument("--timeout", type=int, default="180")
    parser.add_argument("-seed", type=int, default=10000) 
    parser.add_argument("-task", type=str, default="task1") 
    args = parser.parse_args()
    logger.info(f'Input arguments: {args}')
    time.sleep(2)
    config = ConfigParser()
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    os.environ["OPENAI_API_BASE"] = os.getenv("PROXY")
    # transfer
    task_id = args.task
    data_repo_path = f"./knowledge/{args.db}/repo"
    performance_metric = ['tps'] # tps or -lat
    # set path
    target_knobs_path = f"./knowledge/{args.db}/key_knobs_{args.test}_{task_id}.txt"
    suggested_knobs_path = f"./knowledge/{args.db}/suggested_knobs_value/suggested_knobs_value_{args.test}_{task_id}.json" # need to change
    vector_store_path = f"./knowledge/{args.db}/vector_store_chroma"
    to_vector_store_path = f"./knowledge/{args.db}/to_vector_store"
    candidate_knobs_path = f"./knowledge/{args.db}/all_knobs.txt"
    
    init_configs_path = f"./knowledge/{args.db}/init_configs_{args.test}.json"
    init_configs_perfs_path = f"./knowledge/{args.db}/init_configs_perfs_{args.test}.json"
    incumbents_transfer_path = f"./knowledge/{args.db}/incumbents_transfer_{args.test}.json"
    extra_knobs_configs_path = f"./knowledge/mysql/extra_knobs_configs_{args.test}.json"
    rag_method = "graphRAG"
    version = "8.0"


    if args.db == 'postgres':
        config_path = "./configs/postgres.ini"
        config.read(config_path)
        dbms = PgDBMS.from_file(config)
    elif args.db == 'mysql':
        config_path = "./configs/mysql.ini"
        config.read(config_path)
        dbms = MysqlDBMS.from_file(config)
    else:
        raise ValueError("Illegal dbms!")
    
    # store the optimization results
    folder_path = "../optimization_results/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)  

    # init
    if os.path.exists(init_configs_perfs_path):
        logger.info("Already get the init_configs_perfs")
    else:
        ragtuner_init = InitStage(
            dbms=dbms, 
            target_knobs_path=candidate_knobs_path, 
            skill_path=init_configs_path, 
            test=args.test, 
            timeout=args.timeout, 
            init_configs_perfs=init_configs_perfs_path,
            seed=args.seed,
        )
        ragtuner_init.optimize(
            vector_store_path=vector_store_path, 
            init_number=10
        )

    # Transfer
    if not os.path.exists(incumbents_transfer_path) or not os.path.exists(target_knobs_path):
        transfer = Transfer(
            task_id=task_id, 
            data_repo_path=data_repo_path, 
            candidate_knobs_path=candidate_knobs_path,
            performance_metric=performance_metric,
            dbms=dbms,
            init_configs_perfs_path = init_configs_perfs_path,
            test=args.test,
            extra_knobs_configs_path=extra_knobs_configs_path
        )
        similar_task, weights = transfer.get_similary_task()
        if not os.path.exists(incumbents_transfer_path):
            transfer.get_incumbents_transfer(incumbents_transfer_path)

        # Select target knobs
        knob_selector = KnobSelection(
            dbms=dbms,
            benchmark=args.test,
            vector_store_path=vector_store_path,
            version=version,
            candidate_knobs_path=candidate_knobs_path,
            target_knobs_path=target_knobs_path,
            similar_task=similar_task,
            weights=weights,
            key_knobs_num = 15,
            append_ratio = 3,
            rag_method=rag_method # prompt, selfRAG, graphRAG
        )
    
    # prepare structured knowledge

    if not os.path.exists(suggested_knobs_path):
        suggest_rag = SuggestRAG(dbms=dbms, benchmark=args.test, vector_store_path=vector_store_path, target_knobs_path=target_knobs_path, history_transfer_path=incumbents_transfer_path, suggested_knobs_path=suggested_knobs_path, rag_method=rag_method)
        if not os.path.exists(vector_store_path):
            # persist the vector store if no vector store
            suggest_rag.persist(to_vector_store_path, vector_store_path)
        # load data
        suggest_rag.start()
    

    from config_recommender.first_stage import FirstStage
    ragtuner_first = FirstStage(
        dbms=dbms, 
        target_knobs_path=target_knobs_path, 
        skill_path=suggested_knobs_path,
        incumbents_transfer_path=incumbents_transfer_path,
        extra_knobs_configs_path= extra_knobs_configs_path,
        test=args.test, 
        timeout=args.timeout, 
        seed=args.seed,
        objective=performance_metric[0],
        dir_path = f"./optimization_results/{args.db}/first/{args.seed}/"
    )
    number = ragtuner_first.tune_end2end(30, 5)

    from config_recommender.second_stage import SecondStage
    ragtuner_second = SecondStage(
        dbms=dbms, 
        target_knobs_path=target_knobs_path, 
        skill_path=suggested_knobs_path,
        incumbents_transfer_path=incumbents_transfer_path,
        extra_knobs_configs_path= extra_knobs_configs_path,
        test=args.test, 
        timeout=args.timeout, 
        seed=args.seed,
        objective=performance_metric[0],
        dir_path = f"./optimization_results/{args.db}/second/{args.seed}/",
        extra_knob_space=extra_knobs_configs_path
    )
    ragtuner_second.tune_end2end(100-number)

