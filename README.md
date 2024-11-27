# Rabbit


Rabbit is a database tool for knob tuning with RAG.

## 1. Environment Setup

To set up the environment for Rabbit, follow these steps:

1. Create a conda virtual environment named `Rabbit`:
   ```bash
   conda create -n Rabbit python=3.10
   ```

2. Activate the virtual environment:
   ```bash
   conda activate Rabbit
   ```

3. Install the required dependencies using `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```


4. Make a copy of the example environment variables file and modify the OPENAI_API_KEY, PROXY, LLM_MODEL:

   ```bash
   cp .env.template .env
   ```

5. **Add your API key to the newly created .env file.**

Now, you have successfully set up the environment for Rabbit. You can proceed with using the tool for knob tuning with RAG.

## 2. Benchmark

1. Install `PostgreSQL` and `MySQL`

2. Install [BenchBase](https://github.com/cmu-db/benchbase) with our `scripts/`
   
   ```bash
   cd ./scripts
   sh install_benchbase.sh mysql
   ```

3. Set up specific benchmark (e.g., TPC-C).

Note: modify `./benchbase/target/benchbase-mysql/config/mysql/sample_{your_target_benchmark}_config.xml` to customize your tuning setting first

   ```bash
   sh build_benchmark.sh mysql tpcc
   ```

## 3. Using Rabbit

1. modify configs/mysql.ini to determine the target DBMS first, the restart and recover commands depend on the environment 

2. Structure the document information into a diagram and generate a knob knowledge graph

Note: The input document needs to be placed in `./knowledge/mysql/graph/input/` Modify `./knowledge/mysql/graph/settings.yaml` for your enviroment and run:
   
   ```bash
   sh build_graph.sh
   ```
The graph knowledge will output in `./knowledge/mysql/graph/output/`.



3. Run Rabbit like this:
   ```bash
   python run.py
   ```

4. Observe the output of the command line, and its results will exist in the `optimization_results/` folder.

