import os

queries_dir = './88_queries_out'
output_file = './tpch_88.sql'

# Get all .sql files and sort them naturally
sql_files = [f for f in os.listdir(queries_dir) if f.endswith('.sql')]
# Sort by variant then query number: q1_1, q2_1, q3_1, ... q22_1, q1_2, q2_2, ...
# Extracts numbers from 'qX_Y.sql' format to avoid alphabetical sorting (q10 before q2)
sql_files.sort(key=lambda x: (int(x.split('_')[1].split('.')[0]), int(x.split('_')[0][1:])))
# print sql_files names
print("Sorted SQL files:")
for f in sql_files:
    print(f)
with open(output_file, 'w') as out:
    for sql_file in sql_files:
        file_path = os.path.join(queries_dir, sql_file)
        with open(file_path, 'r') as f:
            query = f.read()
            # Remove seed comments
            lines = [line for line in query.split('\n') if not line.strip().startswith('-- using')]
            query = '\n'.join(lines)
            # Remove all whitespace and newlines, then collapse multiple spaces
            query = ' '.join(query.split())
            out.write(query + '\n')

print(f"Created {output_file} with {len(sql_files)} queries (each on single line)")
