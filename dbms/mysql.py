from dbms.dbms_template import DBMSTemplate
import mysql.connector
import os
import json
import time
from util.logger_config import logger

class MysqlDBMS(DBMSTemplate):
    """ Instantiate DBMSTemplate to support PostgreSQL DBMS """
    def __init__(self, db, user, password, restart_cmd, recover_script, knob_info_path):
        super().__init__(db, user, password, restart_cmd, recover_script, knob_info_path)
        self.name = "mysql"
        # self.global_vars = [t[0] for t in self.query_all(
        #     'show global variables') if self.is_numerical(t[1])]
        self.global_vars = [t[0] for t in self.query_all(
            'show global variables') ]
        self.server_cost_params = [t[0] for t in self.query_all(
            'select cost_name from mysql.server_cost')]
        self.engine_cost_params = [t[0] for t in self.query_all(
            'select cost_name from mysql.engine_cost')]
        self.all_variables = self.global_vars + \
            self.server_cost_params + self.engine_cost_params
    
    def _connect(self, db=None):
        self.failed_times = 0
        if db==None:
            db=self.db
        logger.info(f'Trying to connect to {db} with user {self.user}')
        while True:
            try:
                self.connection = mysql.connector.connect(
                    database=db,
                    user=self.user,
                    password=self.password,
                    host="localhost"
                )
                logger.info(f"Success to connect to {db} with user {self.user}")
                return True
            except Exception as e:
                self.failed_times = 4
                logger.info(f'Exception while trying to connect: {e}')
                if self.failed_times <= 4:
                    self.recover_dbms()
                    print("Reconnet again")
                else:
                    return False
                time.sleep(3)

            
    def _disconnect(self):
        if self.connection:
            logger.info('Disconnecting ...')
            self.connection.close()
            logger.info('Disconnecting done ...')
            self.connection = None
    
    def copy_db(self, source_db, target_db):
        ms_clc_prefix = f'mysql -u{self.user} -p{self.password} '
        ms_dump_prefix = f'mysqldump -u{self.user} -p{self.password} '
        os.system(ms_dump_prefix + f' {source_db} > copy_db_dump')
        logger.info('Dumped old database')
        os.system(ms_clc_prefix + f" -e 'drop database if exists {target_db}'")
        logger.info('Dropped old database')
        os.system(ms_clc_prefix + f" -e 'create database {target_db}'")
        logger.info('Created new database')
        os.system(ms_clc_prefix + f" {target_db} < copy_db_dump")
        logger.info('Initialized new database')

    def query_one(self, sql):
        try:
            cursor = self.connection.cursor(buffered=True)
            cursor.execute(sql)
            return cursor.fetchone()[0]
        except Exception:
            return None
        
    def query_one_value(self, sql):
        try:
            cursor = self.connection.cursor(buffered=True)
            cursor.execute(sql)
            return cursor.fetchone()
        except Exception:
            return None
    
    def query_all(self, sql):
        try:
            cursor = self.connection.cursor(buffered=True)
            cursor.execute(sql)
            results = cursor.fetchall()
            cursor.close()
            return results
        except Exception as e:
            logger.info(f'Exception in mysql.query_all: {e}')
            return None
    
    def reset_config(self):
        """ Reset all parameters to default values. """
        self._disconnect()

        self.restart_dbms()
        time.sleep(2)
        res= False
        while not res:
            logger.info("Reconnecting for reconfiguring...")
            res = self._connect()
        self.update_dbms('update mysql.server_cost set cost_value = NULL')
        self.update_dbms('update mysql.engine_cost set cost_value = NULL')
        self.config = {}

    def reconfigure(self):
        """Makes all parameter changes take effect"""
        self.update_dbms('flush optimizer_costs')
        self._disconnect()
        success = self._connect()
        if success:
            return success
        else:
            try:
                self.recover_dbms()
                time.sleep(3)
                return True
            except Exception as e:
                logger.info(f'Exception while trying to recover dbms: {e}')
                return False

    def update_dbms(self, sql):
        """ Execute sql query on dbms to update knob value and return success flag """
        try:
            self.connection.autocommit = True
            cursor = self.connection.cursor(buffered=True)
            cursor.execute(sql, multi=True)
            cursor.close()
            return True
        except Exception as e:
            logger.info(f"Failed to execute {sql} to update dbms for error: {e}")
            return False 

    def extract_knob_info(self, dest_path):
        """ Extract knob information and store the query result in json format """
        knob_info = {}
        knobs_sql = "SHOW VARIABLES;"
        knobs, _ = self.get_sql_result(knobs_sql)
        for knob in knobs:
            knob = knob[0] 
            knob_details_sql = f"SHOW VARIABLES WHERE VARIABLE_NAME = '{knob}';"
            knob_detail, description = self.get_sql_result(knob_details_sql)
            # logger.info(knob, knob_detail)
            if knob_detail:
                column_names = [desc[0] for desc in description]
                knob_detail = knob_detail[0]
                knob_attributes = {}
                for i, column_name in enumerate(column_names):
                    knob_attributes[column_name] = knob_detail[i]
                knob_info[knob] = knob_attributes
            logger.info(f"There are {len(knob_info)} knobs extracted.")
        with open(dest_path, 'w') as json_file:
            json.dump(knob_info, json_file, indent=4, sort_keys=True, default=self.datetime_serializer)
        logger.info(f"The knob info is written to {dest_path}.")

    def get_sql_result(self, sql):
        """ Execute sql query on dbms and return the result and its description """
        self.connection.autocommit = True
        cursor = self.connection.cursor()
        cursor.execute(sql)
        result = cursor.fetchall()
        description = cursor.description
        cursor.close()
        return result, description

    def set_knob(self, knob, knob_value):
        
        if knob in self.global_vars:
            success = self.update_dbms(f'set global {knob}={knob_value}')
            logger.info(f'set global {knob}={knob_value}')
        elif knob in self.server_cost_params:
            success = self.update_dbms(
                f"update mysql.server_cost set cost_value={knob_value} where cost_name='{knob}'")
            logger.info(f"update mysql.server_cost set cost_value={knob_value} where cost_name='{knob}'")
        elif knob in self.engine_cost_params:
            success = self.update_dbms(
                f"update mysql.engine_cost set cost_value={knob_value} where cost_name='{knob}'")
            logger.info(f"update mysql.engine_cost set cost_value={knob_value} where cost_name='{knob}'")
        else: 
            success = False
        if success:
            if knob in self.global_vars:
                v = self.query_one(f"SELECT VARIABLE_VALUE FROM performance_schema.global_variables WHERE VARIABLE_NAME = '{knob}';")
            elif knob in self.server_cost_params:
                v = self.query_one(f"select cost_value from mysql.server_cost where cost_name = '{knob}';")
            elif knob in self.engine_cost_params:
                v = self.query_one(f"select cost_value from mysql.server_cost where cost_name = '{knob}';")
            else:
                v = self.query_one(f"select variables like '{knob}';")
            self.config[knob] = v
        else: 
            self.config[knob] = self.knob_info[knob]["reset_val"]
        return success

    def write_to_reset_knob_config_and_check(self, cnf_file, target_knobs, config):
        """
        将 MySQL 的参数写入配置文件，覆盖写入整个文件。
        参数:
        cnf_file (str): 配置文件的路径。
        target_knobs (list): 配置旋钮列表。
        config: 旋钮的配置值
        """
        self._disconnect()
        self.config = {}
        sc = {}
        ec = {}

        try:
            with open(cnf_file, 'w') as file:
                # 写入固定的前四行
                file.write('!includedir /etc/mysql/conf.d/\n')
                file.write('!includedir /etc/mysql/mysql.conf.d/\n')          
                file.write('\n[mysqld]\n')
                file.write('port = 3306\n')

                if target_knobs is not None and config is not None:
                    # 从第五行开始写入参数
                    for knob_name in target_knobs:
                        if knob_name in self.global_vars :
                            file.write(f"{knob_name} = {config[knob_name]}\n")
                        elif knob_name in self.server_cost_params:
                            sc[knob_name] = config[knob_name]
                        elif knob_name in self.engine_cost_params:
                            ec[knob_name] = config[knob_name]
                        else:
                            file.write(f"{knob_name} = {config[knob_name]}\n")

            logger.info(f"Configuration written to {cnf_file}")
        except IOError as e:
            logger.info(f"Failed to write to config file {cnf_file}: {e}")
        
        self.restart_dbms()
        time.sleep(2)
        res = self._connect()
        if res:
            self.update_dbms('update mysql.server_cost set cost_value = NULL')
            self.update_dbms('update mysql.engine_cost set cost_value = NULL')
            self.update_dbms('flush optimizer_costs')

            for knob_name, knob_value in sc.items():
                self.update_dbms(f"update mysql.server_cost set cost_value={knob_value} where cost_name='{knob_name}'")
            for knob_name, knob_value in ec.items():
                self.update_dbms(f"update mysql.server_cost set cost_value={knob_value} where cost_name='{knob_name}'")

            if target_knobs is not None and config is not None:
                self.check_knob_change(target_knobs, config)
            return True

        else:
            try:
                with open("/etc/mysql/my.cnf", 'w') as file:
                # 写入固定的前四行
                    file.write('!includedir /etc/mysql/conf.d/\n')
                    file.write('!includedir /etc/mysql/mysql.conf.d/\n')          
                    file.write('\n[mysqld]\n')
                    file.write('port = 3306\n')
                self.recover_dbms()
                time.sleep(3)
            except Exception as e:
                logger.info(f'Exception while trying to recover dbms: {e}')
            self.failed_times = 4
            return False
        
    def check_knob_change(self, target_knobs, config):
        #检查knob设置是否成功
        for knob_name in target_knobs:
            
            if knob_name in self.global_vars:
                v = self.query_one(f"SELECT VARIABLE_VALUE FROM performance_schema.global_variables WHERE VARIABLE_NAME = '{knob_name}';")
            elif knob_name in self.server_cost_params:
                v = self.query_one(f"select cost_value from mysql.server_cost where cost_name = '{knob_name}';")
            elif knob_name in self.engine_cost_params:
                v = self.query_one(f"select cost_value from mysql.server_cost where cost_name = '{knob_name}';")
            else:
                v = self.query_one(f"select variables like '{knob_name}';")
                
            self.config[knob_name] = config[knob_name]
            if str(v) == str(config[knob_name]):
                
                logger.info(f"{knob_name} = '{config[knob_name]}' set OK.")
            else:
                logger.info(f"{knob_name} = '{config[knob_name]}' set failed, currnet value is {v} .")


    def get_knob_value(self, knob):
        cursor = self.connection.cursor()
        cursor.execute(f'SHOW VARIABLES LIKE "{knob}"')
        result = cursor.fetchone()
        cursor.close()
        return result

    def check_knob_exists(self, knob):
        cursor = self.connection.cursor()
        cursor.execute(f"SHOW VARIABLES LIKE '{knob}'")
        row = cursor.fetchone()
        cursor.close()
        return row is not None

    def exec_quries(self, sql):
        """ Executes all SQL queries in given file and returns success flag. """
        try:
            self.connection.autocommit = True
            cursor = self.connection.cursor()
            sql_statements = sql.split(';')
            for statement in sql_statements:
                if statement.strip():
                    cursor.execute(statement)
            cursor.close()
            return True
        except Exception as e:
            logger.info(f'Exception execution {sql}: {e}')
        return False
    
    def get_data_size(self):
        dbname = "benchbase"
        sql = 'SELECT CONCAT(round(sum((DATA_LENGTH + index_length) / 1024 / 1024), 2), "MB") as data from information_schema.TABLES where table_schema="{}"'.format(dbname)
        res = self.query_all(sql)
        db_size = float(res[0][0][:-2])
        return db_size
    
    def get_tables_info(self):
        try:
            self.connection.autocommit = True
            cursor = self.connection.cursor()
            cursor.execute("""
                SELECT table_name, table_rows
                FROM information_schema.tables
                WHERE table_schema = DATABASE()
            """)
            tables = cursor.fetchall()

            table_rows = [f"{table_name}({row_count})" for table_name, row_count in tables]
            cursor.close()
            return ",".join(table_rows)
        except Exception as e:
            logger.info(f'Exception execution select count(*): {e}')
            return None
