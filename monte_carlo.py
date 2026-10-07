from statistics import fmean, stdev

from scenarios.runner import run_scenario

SCENARIO = "scenarios/cold_morning.yaml"
NUMBER_OF_RUNS = 100

results = [
    run_scenario(SCENARIO, seed=seed, save_outputs=False)
    for seed in range(NUMBER_OF_RUNS)
]

for metric in results[0]:
    values = [result[metric] for result in results]
    standard_deviation = stdev(values) if len(values) > 1 else 0.0

    print(
        f"{metric}: mean={fmean(values):.3f}, "
        f"std={standard_deviation:.3f}, "
        f"min={min(values):.3f}, max={max(values):.3f}"
    )