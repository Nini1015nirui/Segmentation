#!/usr/bin/env python3
import os
from nnunetv2.experiment_planning.plan_and_preprocess_api import (
    extract_fingerprints, plan_experiments, preprocess,
)

def main():
    ds = [804]
    # 1) Extract fingerprints with dataset integrity check and clean
    extract_fingerprints(ds, check_dataset_integrity=True, clean=True, verbose=True)
    # 2) Plan with reduced GPU memory target and custom plans identifier
    plan_experiments(ds, gpu_memory_target_in_gb=4, overwrite_plans_name='nnUNetPlans_mem4')
    # 3) Preprocess only 2d using the custom plans identifier
    preprocess(ds, plans_identifier='nnUNetPlans_mem4', configurations=['2d'], num_processes=[4], verbose=False)
    print('Done: preprocess 2d (plans: nnUNetPlans_mem4)')

if __name__ == '__main__':
    main()

