from datetime import datetime
from configparser import ConfigParser
import threading
import psycopg2
import time
import re

def generate_random_string():
    return datetime.now().strftime("%m%d_%H_%M_%S")

class one_thread_given_queries(threading.Thread):
    def __init__(self, wg, connection, cur, thread_id, time_stamp, stop_event):
        threading.Thread.__init__(self)
        self.wg = wg
        self.connection = connection
        self.cur = cur
        self.thread_id = thread_id
        self.time_stamp = time_stamp
        self.stop_event = stop_event

    def run(self):
        try:
            start_time = time.time()
            for i, sql in enumerate(self.wg):
                single_start = time.time()
                if self.stop_event.is_set():
                    print(f"Thread {self.thread_id} stopped by timeout")
                    break
                
                self.cur.execute(sql)
                self.connection.commit()
                single_end = time.time()
                print(f"Thread {self.thread_id}: Query {i+1} executed in {single_end - single_start:.2f}s")
            
            end_time = time.time()
            self.time_stamp[self.thread_id] = (len(self.wg), end_time - start_time)
            
        except Exception as e:
            print(f"Thread {self.thread_id} error: {e}")

class ManualWorkloadRunner:
    def __init__(self, config_path=''):
        config_path = '/home/karimnazarovj/Rabbit/configs/postgres.ini'
        config = ConfigParser()
        config.read(config_path)    
        
        # Database config
        self.db_name = config.get('DATABASE', 'db')
        self.db_user = config.get('DATABASE', 'user')
        self.db_password = config.get('DATABASE', 'password')
        self.db_host = config.get('DATABASE', 'host', fallback='localhost')
        self.db_port = config.getint('DATABASE', 'port', fallback=5432)
        
        # Workload config
        self.wg_path = config.get('WORKLOAD', 'workload_path')
        self.thread_num = config.getint('WORKLOAD', 'threads')
        self.timeout = config.getint('WORKLOAD', 'timeout', fallback=600)
        
        self.id = generate_random_string()
        self.sql_list_idx = {}
        self.stop_event = threading.Event()

        print(f'Database Config: {self.db_name} with user {self.db_user} on {self.db_host}:{self.db_port}')

    def get_connection(self):
        """Create new DB connection"""
        conn = psycopg2.connect(
            database=self.db_name,
            user=self.db_user,
            password=self.db_password,
            host=self.db_host,
            port=self.db_port
        )
        return conn, conn.cursor()

    def cancel_all_queries(self):
        """Terminate all active queries on timeout"""
        try:
            conn, cur = self.get_connection()
            cur.execute("""
                SELECT pg_terminate_backend(pid) 
                FROM pg_stat_activity 
                WHERE datname = %s 
                  AND state = 'active' 
                  AND pid <> pg_backend_pid()
            """, (self.db_name,))
            conn.close()
            print("All active queries terminated")
        except Exception as e:
            print(f"Error canceling queries: {e}")

    def data_pre(self):
        """Parse SQL file and distribute to threads"""
        with open(self.wg_path, 'r') as f:
            sql_list = [line.strip() for line in f if line.strip() and not line.strip().startswith('--')]
        
        # Round-robin distribution
        for i in range(self.thread_num):
            self.sql_list_idx[i] = []
        
        for i, sql in enumerate(sql_list):
            self.sql_list_idx[i % self.thread_num].append(sql)
        
        print(f"Loaded {len(sql_list)} queries, {self.thread_num} threads")

    def run(self):
        """Execute workload with timeout"""
        threads = []
        time_stamp = {}
        
        # Create threads with individual connections
        for i in range(self.thread_num):
            conn, cur = self.get_connection()
            thread = one_thread_given_queries(
                wg=self.sql_list_idx[i],
                connection=conn,
                cur=cur,
                thread_id=i,
                time_stamp=time_stamp,
                stop_event=self.stop_event
            )
            threads.append(thread)
        
        # Start all threads
        start_time = time.time()
        for t in threads:
            t.start()
        
        # Wait with timeout
        for t in threads:
            remaining = self.timeout - (time.time() - start_time)
            t.join(timeout=max(0, remaining))
        
        # Check timeout
        if time.time() - start_time >= self.timeout:
            print(f"\n⏱️  TIMEOUT ({self.timeout}s) - Canceling queries...")
            self.stop_event.set()
            self.cancel_all_queries()
        
        end_time = time.time()
        total_execution_time = end_time - start_time
        
        # Calculate results
        total_queries = sum(ts[0] for ts in time_stamp.values())
        throughput = total_queries / total_execution_time if total_execution_time > 0 else 0
        
        # Detect failure and apply penalty
        if total_queries == 0 or throughput == 0:
            penalty_time = self.timeout * 2
            print(f"\n{'='*50}")
            print(f"❌ WORKLOAD FAILED - No queries completed!")
            print(f"Applying penalty: throughput=0, time={penalty_time}s")
            print(f"{'='*50}")
            return 0, penalty_time
        
        print(f"\n{'='*50}")
        print(f"Total queries: {total_queries}")
        print(f"Total execution time: {total_execution_time:.2f}s")
        print(f"Throughput: {throughput:.2f} queries/sec")
        print(f"{'='*50}")
        
        return throughput, total_execution_time