import sys
def process_benchmark_file(file_path):
    benchmark_count = 0
    impact_values = []
    probe_values = []
    ori_values = []

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        for line in file:
            line = line.strip()
            if line.startswith('[benchmark]'):  
                benchmark_count += 1
                parts = line.split()
                if 'impact' in parts:
                    impact_index = parts.index('impact') + 1
                    if impact_index < len(parts): 
                        impact_value = float(parts[impact_index])
                        impact_values.append(impact_value)
                probe_index = parts.index('kernel') + 1
                probe_values.append(float(parts[probe_index]))
                ori_index = parts,index('original') + 1
                ori_values.append(float(parts[ori_index]))

    if impact_values:
        average_impact = sum(impact_values) / len(impact_values)
    else:
        average_impact = 0.0

    impact = sum(probe_values) / sum(ori_values)
    print(f"Total benchmarks: {benchmark_count}")
    print(f"Average impact value: {average_impact:.6f}")
    print(f"Average impact value with value: {impact:.6f}")

file_path = sys.argv[1]
process_benchmark_file(file_path)