import sys
import os
import json
import random
import _utils as utils # utils.py is a custom module that should be in the same directory
import subprocess



PYSLAM_ROOT = r"/home/isaac/dev/pyslam" 
DATA_ROOT = r"/home/isaac/dev/pyslam/data"

def write_yaml_file(file_path, data):
    if os.path.exists(file_path):
        os.remove(file_path)

    with open(file_path, "w") as f:
        if isinstance(data, dict):
            json.dump(data, f, indent=4)
        elif isinstance(data, str):
            f.write(data)
        else:
            raise ValueError("Data must be a dictionary or a string.")

def set_options(orb_feats=1000):
    dataset_options = ["TARTANAIR", "KITTI", "EUROC"]
    dataset = random.choice(dataset_options)

    if dataset == "TARTANAIR":
        dataset = os.path.join(DATA_ROOT, "TARTANAIR")
        if not os.path.exists(dataset):
            raise FileNotFoundError(f"Dataset {dataset} not found in {DATA_ROOT}")
        
        subsets = [
            "abandonedfactory_night_sample_P002",
            "neighborhood_sample_P002",
            "abandonedfactory_sample_P001",
            "ocean_sample_P006",
            "amusement_sample_P008",
            "office2_sample_P003",
            "carwelding_sample_P007",
            "seasidetown_sample_P003",
            "endofworld_sample_P001",
            "seasonsforest_sample_P002",
            "gascola_sample_P001",
            "seasonsforest_winter_sample_P006",
            "hospital_sample_P000",
            "soulcity_sample_P003",
            "japanesealley_sample_P007",
            "westerndesert_sample_P002",
        ]
        
        subset = random.choice(subsets)
        dataset = os.path.join(dataset, subset)

        settings = utils.tartanair_settings(orb_feats)


        settings_path = os.path.join(os.getcwd(), "_settings.yaml")
        write_yaml_file(settings_path, settings)

        slam_config = utils.tartanair_config(dataset, settings_path)
        slam_config_path = os.path.join(os.getcwd(), "_slam_config.yaml")
        write_yaml_file(slam_config_path, slam_config)

        print(f"Running TARTANAIR {subset} with {orb_feats} ORB features")


    elif dataset == "KITTI":
        dataset = os.path.join(DATA_ROOT, "KITTI")
        if not os.path.exists(dataset):
            raise FileNotFoundError(f"Dataset {dataset} not found in {DATA_ROOT}")
        
        # note 01, 03, 04, 09 are only sort of working
        subsets = [
            "00",
            "01",
            "02",
            "03",
            "04",
            "05",
            "06",
            "07",
            "08",
            "09",
            "10",
        ]
        
        subset = random.choice(subsets)
        dataset = os.path.join(dataset, subset)

        settings = utils.kitti_settings(orb_feats, subset)

        settings_path = os.path.join(os.getcwd(), "_settings.yaml")
        write_yaml_file(settings_path, settings)

        slam_config = utils.kitti_config(subset, settings_path)
        slam_config_path = os.path.join(os.getcwd(), "_slam_config.yaml")
        write_yaml_file(slam_config_path, slam_config)

        print(f"Running KITTI {subset} with {orb_feats} ORB features")

    elif dataset == "EUROC":
        dataset = os.path.join(DATA_ROOT, "EuRoC")
        if not os.path.exists(dataset):
            raise FileNotFoundError(f"Dataset {dataset} not found in {DATA_ROOT}")
        
        subsets = [
            "MH01",
            "MH02",
            "MH03",
            "MH04",
            "MH05",
            "V101",
            "V102",
            "V103",
            "V201",
            "V202",
            "V203",
        ]
        
        subset = random.choice(subsets)
        dataset = os.path.join(dataset, subset)

        settings = utils.euroc_settings(orb_feats)

        settings_path = os.path.join(os.getcwd(), "_settings.yaml")
        write_yaml_file(settings_path, settings)

        slam_config = utils.euroc_config(subset, settings_path)
        slam_config_path = os.path.join(os.getcwd(), "_slam_config.yaml")
        write_yaml_file(slam_config_path, slam_config)

        print(f"Running EUROC {subset} with {orb_feats} ORB features")

def run_slam():
    # run the run.bash script
    run_script = os.path.join(os.getcwd(), "_run.bash")
    if not os.path.exists(run_script):
        raise FileNotFoundError(f"Run script {run_script} not found")
    
    res = subprocess.run(["bash", run_script], capture_output=True, check=True)
    
    if res.returncode != 0:
        print(f"Error running the script: {res.stderr}")

def check_results():
    res_dir = r"/home/isaac/dev/pyslam/results"
    results = os.listdir(res_dir)
    try:
        result_dir = os.path.join(res_dir, results[0]) # load the first result (we are deleting all others)
        stats_file = os.path.join(result_dir, "plot", "stats_final.json")
        with open(stats_file, "r") as f:
            stats = json.load(f)

        ret = stats["rmse"]
    except Exception as e:
        print(f"Error reading the results: {e}")
        ret = None

    # remove all folders in the results directory
    for result in results:
        result_path = os.path.join(res_dir, result)
        if os.path.isdir(result_path):
            subprocess.run(["rm", "-rf", result_path])
    return ret

if __name__ == "__main__":
    possible_orb_feats = [600, 1000, 3000]
    for feats in possible_orb_feats:
        set_options(feats)
        run_slam()
        rmse = check_results()
        print(f"RMSE: {rmse}")