import argparse
import subprocess
import yaml


def edit_config(base_config, vary_tuple, vary_key): 
    list_of_zarr = base_config['task']['dataset']['zarr_configs']
    edited_list_of_zarr = []
    for zarr_dict in list_of_zarr: 
        path= zarr_dict['path']
        components = path.split('/')
        last_component = components[-1]

        if 'real' in last_component: 
            zarr_dict[vary_key] = vary_tuple[0]
        elif 'sim' in last_component:
            zarr_dict[vary_key] = vary_tuple[1]
        else: 
            raise ValueError('real or sim not in zarr path')
        edited_list_of_zarr.append(zarr_dict)
    base_config['task']['dataset']['zarr_configs'] = edited_list_of_zarr
    return base_config

def main(base_config, mixed_list, vary_key, config_dir): 
    base_command = ("train.py", [f"--config-dir={config_dir}", "--config-name=temp_for_sweep_run.yaml"])

    scripts_with_args = []
    for i in range(len(mixed_list)): 
        scripts_with_args.append(base_command)

    for i, (script, args) in enumerate(scripts_with_args): 
        new_base_config = edit_config(base_config, mixed_list[i], vary_key)
        with open(f'{config_dir}/temp_for_sweep_run.yaml', 'w') as f: 
            yaml.dump(new_base_config, f)

        result = subprocess.run(['python', script] + args)
        if result.returncode != 0:
            print(f"Error: {script} exited with return code {result.returncode}")
            break
        else:
            print(f"{script} finished successfully.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--study_num', action='store_true', default=False)
    parser.add_argument('--study_scale', action='store_true', default=False)
    parser.add_argument('--num_real', type=int, default=0)
    parser.add_argument('--num_sim', type=int, default=0)
    parser.add_argument('--config_dir', type=str, default='config')
    parser.add_argument('--config_name', type=str, default='config.yaml')

    real_data_weights = [0.25, 0.5, 0.75]

    real_data_num = [1, 2, 3]
    sim_data_num = [2, 3]

    args = parser.parse_args()

    if args.study_scale: 
        assert args.num_real > 0 and args.num_sim > 0

    with open(f'{args.config_dir}/{args.config_name}.yaml', 'r') as f:
        base_config = yaml.safe_load(f)

    if args.study_num: 
        mixed_list = []
        for r in real_data_num:
            for s in sim_data_num:
                mixed_list.append((r, s))
            
        main(base_config, mixed_list, 'max_train_episodes', args.config_dir)

    elif args.study_scale: 
        mixed_list = []
        for r in real_data_weights:
            mixed_list.append((r, 1-r))

        list_of_zarr = base_config['task']['dataset']['zarr_configs']
        edited_list_of_zarr = []
        for zarr_dict in list_of_zarr: 
            path= zarr_dict['path']
            components = path.split('/')
            last_component = components[-1]

            if 'real' in last_component:
                zarr_dict['max_train_episodes'] = args.num_real
            elif 'sim' in last_component:
                zarr_dict['max_train_episodes'] = args.num_sim

            edited_list_of_zarr.append(zarr_dict)

        base_config['task']['dataset']['zarr_configs'] = edited_list_of_zarr

        main(base_config, mixed_list, 'sampling_weight', args.config_dir)

