import json
import glob
import sys
from typing import Dict, List, Any

def convert_string(value: str):
    # 尝试转换为整型
    try:
        return int(value)
    except ValueError:
        pass
    
    # 尝试转换为浮点型
    try:
        return float(value)
    except ValueError:
        pass
    
    # 无法转换为整型或浮点型时，返回原始字符串
    return value

def convert_json_files(metric: str) -> None:
    # 获取当前文件夹下所有名称中包含 'history' 的 JSON 文件
    json_files: List[str] = glob.glob('*history*.json')
   
    for json_file in json_files:
        with open(json_file, 'r') as file:
            original_data: Dict[str, Any] = json.load(file)

        # 创建目标数据结构
        target_data: Dict[str, Any] = {
            "info": {
                "objs": [metric],
                "constraints": []
            },
            "data": []
        }

        src_configs: Dict[str, Dict[str, Any]] = original_data.get('configs', {})
        src_data_list: List[List[Any]] = original_data.get('data', [])

        # 遍历原始数据中的配置和data列表，并生成目标数据格式
        for idx, (_, config) in enumerate(src_configs.items()):
            if idx < len(src_data_list):
                value = v if (v:= src_data_list[idx][4]) >= 0 else -v
            else:
                value = 0

            config = {key: convert_string(value) for key, value in config.items()}

            entry: Dict[str, Any] = {
                "configuration": config,
                "external_metrics": {
                    "tps": 0,
                    "lat": 0,
                    "qps": 0,
                    "tpsVar": 0,
                    "latVar": 0,
                    "qpsVar": 0
                },
                "internal_metrics": [],
                "resource": {
                    "cpu": 0,
                    "readIO": 0,
                    "writeIO": 0,
                    "IO": 0,
                    "virtualMem": 0,
                    "physical": 0,
                    "dirty": 0,
                    "hit": 0,
                    "data": 0
                },
                "trial_state": 0,
                "elapsed_time": 200
            }

            # 根据传入的参数动态设置external_metrics的tps或lat字段
            # 这里可加新的种类
            if metric == "-lat":
                entry["external_metrics"]["lat"] = value
            else:
                entry["external_metrics"]["tps"] = value

            target_data['data'].append(entry)

        # 生成的目标文件名
        target_file: str = f"fmted_{json_file.replace('runhistory', 'history')}"
        with open(target_file, 'w') as file:
            json.dump(target_data, file, indent=2)

        print(f"转换完成，生成的文件为 {target_file}")

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("用法: python convert_runhistory.py <metrics>")
        print("例如: python convert_runhistory.py tps")
        print("tps表示吞吐量,-lat表示延迟")
        sys.exit(1)

    metric: str = sys.argv[1]
    convert_json_files(metric)
