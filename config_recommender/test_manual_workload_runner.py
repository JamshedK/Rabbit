from manual_workload_runner import ManualWorkloadRunner

runner = ManualWorkloadRunner()
runner.data_pre()
throughput, total_time = runner.run()

print(f"\nResults: {throughput:.2f} queries/sec, {total_time:.2f}s total")