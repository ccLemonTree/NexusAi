

# Input data
in_dotenv_needed_models = {
}

in_dotenv_needed_paths = {
    "MODELS_HOME": "./BAGEL-7B",
    "MODELS_OFFLOAD":"./modelsoffload"
}



in_dotenv_needed_params = {
    "DEBUG_MODE": False,
    "IMAGE_COMPRESSION_LEVEL_1_TO_100":60

}



in_files_to_check_in_paths=[
]


#LCX1.05##################################################################
#FILELOADER##############################################################
#########################################################################
debug_mode=False
LCX_APP_NAME="CROSSOS_FILE_CHECK"
in_model_config_file="configmodel.txt"
# --- Helper Functions ---
#dotenv prefixes
PREFIX_MODEL="PATH_MODEL_"
PREFIX_PATH="PATH_NEEDED_"
LOG_PREFIX="CROSSOS_LOG"


##Memhelper START#######################################
import torch
import shutil
import subprocess

cpu = torch.device('cpu')
gpu = None
if torch.cuda.is_available():
    gpu = torch.device(f'cuda:{torch.cuda.current_device()}')
elif torch.backends.mps.is_available():
    gpu = torch.device('mps')
else:
    raise RuntimeError("No GPU device available. Please use a system with CUDA or MPS support.")
#returns VRAM in GB.
def get_free_system_vram_total_free_used(device=None, debug_mode=False):
    total=0
    used=0
    free=0
    if device is None:
        device = gpu
    if device.type == 'mps':
        # MPS doesn't provide detailed memory stats, return a best guess
        bytes_total_available = torch.mps.recommended_max_memory() - torch.mps.driver_allocated_memory()
        free= torch.mps.recommended_max_memory()  / (1024 ** 3)
        used= torch.mps.driver_allocated_memory()  / (1024 ** 3)
        total= bytes_total_available / (1024 ** 3)
    elif device.type == 'cuda':
        num_devices = torch.cuda.device_count()
        if debug_mode:
            print(f"Found {num_devices} CUDA device(s)")

        total_vram_all = 0.0
        used_vram_all = 0.0
        free_vram_all = 0.0

        for i in range(num_devices):
            torch.cuda.set_device(i)  # Switch to device `i`
            device = torch.device(f'cuda:{i}')

            # Get memory stats for the current device
            memory_stats = torch.cuda.memory_stats(device)
            bytes_active = memory_stats['active_bytes.all.current']
            bytes_reserved = memory_stats['reserved_bytes.all.current']
            bytes_free_cuda, bytes_total_cuda = torch.cuda.mem_get_info(device)

            # Calculate memory components
            bytes_inactive_reserved = bytes_reserved - bytes_active
            bytes_total_available = bytes_free_cuda + bytes_inactive_reserved

            # Convert to GB
            loop_used = bytes_active / (1024 ** 3)
            loop_free = bytes_total_available / (1024 ** 3)
            loop_total = bytes_total_cuda / (1024 ** 3)

            # Accumulate across all devices
            total_vram_all += loop_total
            used_vram_all += loop_used
            free_vram_all += loop_free
            if debug_mode:
                # Print per-device stats
                print(f"\nDevice {i} ({torch.cuda.get_device_name(i)}):")
                print(f"  Total VRAM: {loop_total:.2f} GB")
                print(f"  Used VRAM:  {loop_used:.2f} GB")
                print(f"  Free VRAM:  {loop_free:.2f} GB")
        if debug_mode:

            # Print aggregated stats
            print("\n=== Total Across All Devices ===")
            print(f"Total VRAM: {total_vram_all:.2f} GB")
            print(f"Used VRAM:  {used_vram_all:.2f} GB")
            print(f"Free VRAM:  {free_vram_all:.2f} GB")
        free = free_vram_all
        total = total_vram_all   # This is more accurate than used+free
        used = total-free
        """
        try:
            nvidia_smi = shutil.which('nvidia-smi')
            if nvidia_smi:
                try:
                    gpu_info = subprocess.check_output([nvidia_smi, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"], encoding='utf-8').strip()
                    gpu_name, vram_total = gpu_info.split(',')
                    #report.append(f"  Model: {gpu_name.strip()}")
                    total= float(vram_total.strip())/1000
                    # Get current VRAM usage if possible
                    try:
                        gpu_usage = subprocess.check_output([nvidia_smi, "--query-gpu=memory.used", "--format=csv,noheader,nounits"], encoding='utf-8').strip()
                        used=float(gpu_usage.strip())/1000
                        free=total-used
                    except:
                        pass
                except Exception as e:
                    print(f"  Could not query GPU info with nvidia-smi: {str(e)}")
        except:
            pass
        """
    total=round(total, 2)
    free=round(free, 2)
    used=round(used, 2)
    if debug_mode:
        print(f"GPU mem total: {total}, free: {free}, used: {used}")
    return total,free,used
#total,free,used=get_free_system_vram_total_free_used()
#print(f"GPU total: {total}, free: {free}, used: {used}")

import psutil
#returns VRAM in GB.
def get_free_system_ram_total_free_used( debug_mode=False):
    total=0
    used=0
    free=0
    ram = psutil.virtual_memory()
    total= round(ram.total / (1024**3), 2)
    free=round(ram.available / (1024**3), 2)
    used=round(ram.used / (1024**3), 2)
    if debug_mode:
        print(f"RAM total: {total}, free: {free}, used: {used}")
    return total,free,used
#total,free,used=get_free_system_ram_total_free_used()
#print(f"RAM total: {total}, free: {free}, used: {used}")


#returns VRAM in GB.
import psutil
def get_free_system_disk_total_free_used(device=None, debug_mode=False):
    total=0
    used=0
    free=0
    try:
        disk = psutil.disk_usage('/')
        total=round(disk.total / (1024**3), 2)
        free= round(disk.free / (1024**3), 2)
        used=round(disk.used / (1024**3), 2)
    except Exception as e:
        print(f"  Could not get disk info: {str(e)}")
    if debug_mode:
        print(f"disk mem total: {total}, free: {free}, used: {used}")
    return total,free,used


#total,free,used=get_free_system_disk_total_free_used()
#print(f"HDD total: {total}, free: {free}, used: {used}")
##Memhelper END#######################################

import re
import os
from pathlib import Path
from typing import Dict, Set, Any, Union
def model_to_varname(model_path: str, prefix: str) -> str:
    """Converts a model path to a dotenv-compatible variable name"""
    model_name = model_path.split("/")[-1]
    varname = re.sub(r"[^a-zA-Z0-9]", "_", model_name.upper())
    return f"{prefix}{varname}"

def varname_to_model(varname: str, prefix: str) -> str:
    """Converts a variable name back to original model path format"""
    if varname.startswith("PATH_MODEL_"):
        model_part = varname[prefix.len():].lower().replace("_", "-")
        return f"Zyphra/{model_part}"
    return ""

def read_existing_config(file_path: str) -> Dict[str, str]:
    """Reads existing config file and returns key-value pairs"""
    existing = {}
    path = Path(file_path)
    if path.exists():
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    parts = line.split("=", 1)
                    if len(parts) == 2:
                        existing[parts[0].strip()] = parts[1].strip()
    else:
        print(f"{LCX_APP_NAME}: ERROR config file not found: {file_path}")
    if debug_mode:
        print(f"{LCX_APP_NAME}: found config file: {file_path}")
    return existing

def update_model_paths_file(
        models: Set[str],
        paths: Dict[str, str],
        params: Dict[str, Any],
        file_path: str
) -> None:
    """Updates config file, adding only new variables"""
    existing = read_existing_config(file_path)
    new_lines = []

    # Process models
    for model in models:
        varname = model_to_varname(model, PREFIX_MODEL)
        if varname not in existing:
            print(f"{LOG_PREFIX}: Adding Model rquirement to config: {model}")
            new_lines.append(f"{varname} = ./models/{model.split('/')[-1]}")

    # Process paths - now handles any path keys
    for key, value in paths.items():
        varname = model_to_varname(key, PREFIX_PATH)
        if varname not in existing:
            print(f"{LOG_PREFIX}: Adding path rquirement to config: {key}")
            new_lines.append(f"{varname} = {value}")

    # Process params
    for key, value in params.items():
        if key not in existing:
            print(f"{LOG_PREFIX}: Adding Parameter rquirement to config: {key}")
            new_lines.append(f"{key} = {value}")

    # Append new lines if any
    if new_lines:
        with open(file_path, 'a') as f:
            f.write("\n" + "\n".join(new_lines) + "\n")

def parse_model_paths_file(file_path: str , dotenv_needed_models, dotenv_needed_paths ) -> tuple[
    Set[str], Dict[str, str], Dict[str, Union[bool, int, float, str]]
]:
    """Reads config file and returns loaded variables"""
    loaded_models = {}
    loaded_paths = {}
    loaded_params = {}
    loaded_models_values= {}
    existing = read_existing_config(file_path)

    for key, value in existing.items():
        # Handle model paths
        if key.startswith(PREFIX_MODEL):
            for mod in dotenv_needed_models:
                #we find out if the current key value belongs to one of our models
                if key == model_to_varname(mod,PREFIX_MODEL):
                    #if a path has been defined and it exists we use the local path
                    if value and os.path.isdir(value):
                        loaded_models[mod] = value
                    else:
                        #else we use the model id so its downloaded from github later
                        loaded_models[mod] = mod
                    #still we collect the values to show to the user so he knows what to fix in config file
                    loaded_models_values[mod] = value
        # Handle ALL paths (not just HF_HOME)
        elif key.startswith(PREFIX_PATH):
            for mod in dotenv_needed_paths:
                if key == model_to_varname(mod,PREFIX_PATH):
                    loaded_paths[mod] = value
        # Handle params with type conversion
        else:
            if value.lower() in {"true", "false"}:
                loaded_params[key] = value.lower() == "true"
            elif value.isdigit():
                loaded_params[key] = int(value)
            else:
                try:
                    loaded_params[key] = float(value)
                except ValueError:
                    loaded_params[key] = value

    return loaded_models, loaded_paths, loaded_params, loaded_models_values

def is_online_model(model: str,dotenv_needed_models, debug_mode: bool = False) -> bool:
    """Checks if a model is in the online models set."""
    is_onlinemodel = model in dotenv_needed_models
    if debug_mode:
        print(f"Model '{model}' is online: {is_onlinemodel}")
    return is_onlinemodel

import os
def count_existing_paths(paths):
    """
    Checks if each path in the list exists.
    Returns:
        - summary (str): Summary of found/missing count
        - all_found (bool): True if all paths were found
        - none_found (bool): True if no paths were found
        - details (list of str): List with "[found]" or "[not found]" per path
    """
    total = len(paths)
    if total == 0:
        return "No paths provided.", False, True, []
    found_count = 0
    details = []
    for path in paths:
        if os.path.exists(path):
            found_count += 1
            details.append(f"[!FOUND!]: {path}")
        else:
            details.append(f"[MISSING]: {path}")
    missing_count = total - found_count
    all_found = (missing_count == 0)
    none_found = (found_count == 0)
    summary = f"Found {found_count}, missing {missing_count}, out of {total} paths."
    return summary, all_found, none_found, details


def remove_suffix(text, suffix):
    if text.endswith(suffix):
        return text[:-len(suffix)]
    return text


def get_hf_model_cache_dirname(model_id: str) -> str:
    """
    Returns the HF cache directory name for a given model.
    """
    base = "models--"
    return base + model_id.replace('/', '--')

def check_do_all_files_exist(dotenv_needed_models,dotenv_loaded_paths, dotenv_loaded_models, dotenv_loaded_models_values, in_files_to_check_in_paths=None, silent=False  ):
    test_models_hf = []
    test_models_dir=[]
    test_paths_dir=[]

    retval_models_exist=True
    retval_paths_exist=True

    #add model paths as path and as hf cache path
    for currmodel in dotenv_needed_models:
        test_models_hf.append(f"{dotenv_loaded_paths['HF_HOME']}{os.sep}hub{os.sep}{get_hf_model_cache_dirname(currmodel)}{os.sep}snapshots")
        test_models_dir.append(f"{dotenv_loaded_models[  currmodel]}")

    #add needed dirs as path
    for curr_path in dotenv_loaded_paths:
        test_paths_dir.append(f"{dotenv_loaded_paths[  curr_path]}")

    if debug_mode:
        print(f"test pathf hf: {test_models_hf}")
        print(f"test pathf dirs: {test_models_dir}")

    if not silent:
        print(f"{LCX_APP_NAME}: checking model accessibility")
    summary_hf, all_exist_hf, none_exist_hf, path_details_hf = count_existing_paths(test_models_hf)

    if not silent:
        print(f"\n-Searching Group1: Model HF_HOME----------------------------------------------")
        for line in path_details_hf:
            print_line= remove_suffix(line, "snapshots")
            print(print_line)

    summary_dir, all_exist_dir, none_exist_dir, path_details_dir = count_existing_paths(test_models_dir)
    if not silent:
        print("-Searching Group2: Manual Model Directories-----------------------------------")
        for line in path_details_dir:
            print_line= remove_suffix(line, "model_index.json")
            print_line= remove_suffix(print_line, "config.json")
            print(print_line)

    summary_path, all_exist_path, none_exist_path, path_details_path = count_existing_paths(test_paths_dir)
    if not silent:
        print("-Searching Group3: Needed Directories-----------------------------------------")
        for line in path_details_path:
            print(line)

    if not silent:
        print("-checking explicite Files---------------------------------------------------")

    for mapping in in_files_to_check_in_paths:
        for env_var, relative_path in mapping.items():
            if dotenv_loaded_paths and env_var in dotenv_loaded_paths:
                base_path = dotenv_loaded_paths[env_var]
                full_path = Path(base_path) / relative_path.strip(os.sep)
                if full_path.exists():
                    if not silent:
                        print(f"[!FOUND!]: {full_path}")
                else:
                    if not silent:
                        print(f"[!MISSING!]: {full_path}")
                    retval_paths_exist = False
    if not silent:
        print("")
    #we show the dir values to the user
    if not silent:
        if all_exist_dir==False:
            print("-Values in config (resolved to your OS)---------------------------------------")
            for key in dotenv_loaded_models_values:
                print(f"{key}: {os.path.abspath(dotenv_loaded_models_values[key])}")
        if all_exist_path==False:
            for key in dotenv_loaded_paths:
                print(f"{key}: {os.path.abspath(dotenv_loaded_paths[  key])}")
    if not silent:
        print("")

    #Needed Dirs summary
    if in_dotenv_needed_paths and not silent:
        print("-Needed Paths---------------------------------------------------")
    if in_dotenv_needed_paths and all_exist_path == False:
        if not silent:
            print("Not all paths were found. Check documentation if you need them")
        retval_paths_exist=False
    if not silent:
        if in_dotenv_needed_paths and all_exist_path:
            print("All Needed PATHS exist.")
    if in_dotenv_needed_models:
        if not silent:
            print("-Needed Models--------------------------------------------------")
            #some model directories were missing
            if none_exist_dir == False and all_exist_dir == False:
                print ("Some manually downloaded models were found. Some might need to be downloaded!")
            #some hf cache models were missing
            if  all_exist_hf == False and none_exist_hf==False:
                print ("Some HF_Download models were found. Some might need to be downloaded!")
            if none_exist_dir and none_exist_hf:
                print ("No models were found! Models will be downloaded at next app start")

            if all_exist_hf==True or all_exist_dir==True:
                print("RESULT: It seems all models were found. Nothing will be downloaded!")
        if all_exist_hf==False and all_exist_dir==False:
            retval_models_exist=False


    retval_final=retval_models_exist == True and retval_paths_exist ==True

    return retval_final

def lcx_checkmodels(dotenv_needed_models,dotenv_loaded_paths, dotenv_loaded_models, dotenv_loaded_models_values, in_files_to_check_in_paths=None  ):
    check_do_all_files_exist(dotenv_needed_models,dotenv_loaded_paths, dotenv_loaded_models, dotenv_loaded_models_values, in_files_to_check_in_paths=in_files_to_check_in_paths  )
    sys.exit()
### SYS REPORT START##################
import sys
import platform
import subprocess
import os
import shutil
import torch
import psutil
from datetime import datetime

def anonymize_path(path):
    """Replace username in paths with <USER>"""
    if not path:
        return path
    # Handle both Unix and Windows paths
    if path.startswith('/home/'):
        parts = path.split('/')
        if len(parts) > 2:
            parts[2] = '<USER>'
            return '/'.join(parts)
    elif path.startswith('/Users/'):
        parts = path.split('/')
        if len(parts) > 2:
            parts[2] = '<USER>'
            return '/'.join(parts)
    elif path.startswith('C:\\Users\\'):
        parts = path.split('\\')
        if len(parts) > 2:
            parts[2] = '<USER>'
            return '\\'.join(parts)
    return path

def generate_troubleshooting_report(in_model_config_file=None):
    """Generate a comprehensive troubleshooting report for AI/LLM deployment issues."""
    # Create a divider for better readability
    divider = "=" * 80

    # Initialize report
    report = []
    report.append(f"{divider}")
    report.append(f"TROUBLESHOOTING REPORT - Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. Hardware Information
    report.append("HARDWARE INFORMATION")

    # CPU Info
    report.append("\nCPU:")
    report.append(f"  Model: {platform.processor()}")
    try:
        cpu_freq = psutil.cpu_freq()
        report.append(f"  Max Frequency: {cpu_freq.max:.2f} MHz")
        report.append(f"  Cores: Physical: {psutil.cpu_count(logical=False)}, Logical: {psutil.cpu_count(logical=True)}")
    except Exception as e:
        report.append(f"  Could not get CPU frequency info: {str(e)}")

    # RAM Info
    ram = psutil.virtual_memory()
    report.append("\nRAM:")
    report.append(f"  Total: {ram.total / (1024**3):.2f} GB: free: {ram.available / (1024**3):.2f} used: {ram.used / (1024**3):.2f} GB")

    # GPU Info
    report.append("\nGPU:")
    try:
        nvidia_smi = shutil.which('nvidia-smi')
        if nvidia_smi:
            try:
                gpu_info = subprocess.check_output([nvidia_smi, "--query-gpu=name,memory.total", "--format=csv,noheader"], encoding='utf-8').strip()
                gpu_name, vram_total = gpu_info.split(',')
                report.append(f"  Model: {gpu_name.strip()}")
                report.append(f"  VRAM: {vram_total.strip()}")

                try:
                    gpu_usage = subprocess.check_output([nvidia_smi, "--query-gpu=memory.used", "--format=csv,noheader"], encoding='utf-8').strip()
                    report.append(f"  VRAM Used: {gpu_usage.strip()}")
                except:
                    pass
            except Exception as e:
                report.append(f"  Could not query GPU info with nvidia-smi: {str(e)}")
    except:
        pass

    # If torch is available and has CUDA, get GPU info from torch
    try:
        if torch.cuda.is_available():
            report.append("\nGPU Info from PyTorch:")
            for i in range(torch.cuda.device_count()):
                report.append(f"  Device {i}: {torch.cuda.get_device_name(i)}, VRAM: {torch.cuda.get_device_properties(i).total_memory / (1024**3):.2f} GB")
    except:
        pass

    # Disk Space
    report.append("\nDISK:")
    try:
        disk = psutil.disk_usage('/')
        report.append(f"  Total: {disk.total / (1024**3):.2f} GB.  Free: {disk.free / (1024**3):.2f} GB, Used: {disk.used / (1024**3):.2f} GB")
    except Exception as e:
        report.append(f"  Could not get disk info: {str(e)}")

    # 2. Software Information
    report.append(f"\n{divider}")
    report.append("SOFTWARE INFORMATION")

    # OS Info
    report.append("\nOPERATING SYSTEM:")
    report.append(f"  System: {platform.system()}")
    report.append(f"  Release: {platform.release()}")
    report.append(f"  Version: {platform.version()}")
    report.append(f"  Machine: {platform.machine()}")

    # Python Info
    report.append("\nPYTHON:")
    report.append(f"  Version: {platform.python_version()}")
    report.append(f"  Implementation: {platform.python_implementation()}")
    report.append(f"  Executable: {anonymize_path(sys.executable)}")

    # Installed packages
    report.append("\nINSTALLED PACKAGES (pip freeze):")
    try:
        pip_freeze = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], encoding='utf-8')
        report.append(pip_freeze)
    except Exception as e:
        report.append(f"  Could not get pip freeze output: {str(e)}")

    # CUDA Info
    report.append("CUDA INFORMATION:")
    try:
        nvcc_path = shutil.which('nvcc')
        if nvcc_path:
            nvcc_version = subprocess.check_output(['nvcc', '--version'], encoding='utf-8')
            report.append(nvcc_version.strip())
        else:
            report.append("NVCC not found in PATH")
    except Exception as e:
        report.append(f"  Could not get NVCC version: {str(e)}")

    # PyTorch CUDA version if available
    try:
        if 'torch' in sys.modules:
            report.append("\nPYTORCH CUDA:")
            report.append(f"  PyTorch version: {torch.__version__}")
            report.append(f"  CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                report.append(f"  CUDA version: {torch.version.cuda}")
                report.append(f"  Current device: {torch.cuda.current_device()}")
                report.append(f"  Device name: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        report.append(f"  Could not get PyTorch CUDA info: {str(e)}")

    # 3. Model Configuration
    if in_model_config_file:
        report.append(f"\n{divider}")
        report.append("MODEL CONFIGURATION")

        try:
            with open(in_model_config_file, 'r') as f:
                config_content = f.read()
            report.append(f"Content of {anonymize_path(in_model_config_file)}:")
            report.append(config_content)
        except Exception as e:
            report.append(f"\nCould not read model config file {anonymize_path(in_model_config_file)}: {str(e)}")

    # 4. Environment Variables
    report.append(f"\n{divider}")
    report.append("RELEVANT ENVIRONMENT VARIABLES")

    relevant_env_vars = [
        'PATH', 'LD_LIBRARY_PATH', 'CUDA_HOME', 'CUDA_PATH',
        'PYTHONPATH', 'CONDA_PREFIX', 'VIRTUAL_ENV'
    ]

    for var in relevant_env_vars:
        if var in os.environ:
            # Anonymize paths in environment variables
            if var in ['PATH', 'LD_LIBRARY_PATH', 'PYTHONPATH']:
                paths = os.environ[var].split(os.pathsep)
                anonymized_paths = [anonymize_path(p) for p in paths]
                report.append(f"{var}: {os.pathsep.join(anonymized_paths)}")
            else:
                report.append(f"{var}: {anonymize_path(os.environ[var])}")

    # 5. Additional System Info
    report.append(f"\n{divider}")
    report.append("ADDITIONAL SYSTEM INFORMATION")

    try:
        # Check if running in container
        report.append("\nContainer/Virtualization:")
        if os.path.exists('/.dockerenv'):
            report.append("  Running inside a Docker container")
        elif os.path.exists('/proc/1/cgroup'):
            with open('/proc/1/cgroup', 'r') as f:
                if 'docker' in f.read():
                    report.append("  Running inside a Docker container")
                elif 'kubepods' in f.read():
                    report.append("  Running inside a Kubernetes pod")
        # Check virtualization
        try:
            virt = subprocess.check_output(['systemd-detect-virt'], encoding='utf-8').strip()
            if virt != 'none':
                report.append(f"  Virtualization: {virt}")
        except:
            pass
    except Exception as e:
        report.append(f"  Could not check container/virtualization info: {str(e)}")

    # Final divider
    report.append("END OF REPORT")
    report.append(f"{divider}")

    # Join all report lines
    full_report = '\n'.join(report)
    return full_report
####END SYS REPORT########################################################################
# Update the config file
update_model_paths_file(in_dotenv_needed_models, in_dotenv_needed_paths, in_dotenv_needed_params, in_model_config_file)

# Read back the values
out_dotenv_loaded_models, out_dotenv_loaded_paths, out_dotenv_loaded_params , out_dotenv_loaded_models_values= parse_model_paths_file(in_model_config_file, in_dotenv_needed_models,in_dotenv_needed_paths)

if debug_mode:
    print("Loaded models:", out_dotenv_loaded_models)
    print("Loaded models values:", out_dotenv_loaded_models_values)
    print("Loaded paths:", out_dotenv_loaded_paths)
    print("Loaded params:", out_dotenv_loaded_params)

if "HF_HOME" in in_dotenv_needed_paths:
    os.environ['HF_HOME'] = out_dotenv_loaded_paths["HF_HOME"]
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
#os.environ["TOKENIZERS_PARALLELISM"] = "true"
#CORE BLOCK END###############################################################################




######INSERT START########################
model_offload_dir=out_dotenv_loaded_paths["MODELS_OFFLOAD"]
VRAM=80
RAM=12
DISK=10
vramtotal,vramfree,vramused=get_free_system_vram_total_free_used()
ramtotal,ramfree,ramused=get_free_system_ram_total_free_used()
disktotal,diskfree,diskused=get_free_system_disk_total_free_used()
THRESH_VRAM_HIGH_RICHPANTS=32
THRESH_VRAM_LOW_POTATO=12
run_mode=0
max_memory={i: "16GiB" for i in range(torch.cuda.device_count())}
max_memory["cpu"]={f"{RAM}GiB"}
max_memory["disk"]={f"{DISK}GiB"}
if vramfree >= THRESH_VRAM_HIGH_RICHPANTS:
    run_mode=1
elif vramfree < THRESH_VRAM_HIGH_RICHPANTS and vramfree >= THRESH_VRAM_LOW_POTATO:
    runmode=2
elif vramfree < THRESH_VRAM_LOW_POTATO:
    runmode=3
    max_memory=None
######INSERT END########################

#originalblock#################################
import argparse
import sys
parser = argparse.ArgumentParser()
parser.add_argument('--share', action='store_true')
parser.add_argument("--server", type=str, default='0.0.0.0')
parser.add_argument("--port", type=int, required=False)
parser.add_argument("--inbrowser", action='store_true')
parser.add_argument("--output_dir", type=str, default='./outputs')
parser.add_argument("--checkmodels", action='store_true')
parser.add_argument("--integritycheck", action='store_true')
parser.add_argument("--sysreport", action='store_true')

parser.add_argument("--server_name", type=str, default="0.0.0.0")
parser.add_argument("--server_port", type=int, default=10010)
parser.add_argument("--model_path", type=str, default=out_dotenv_loaded_paths["MODELS_HOME"])
parser.add_argument("--mode", type=int, default=2)
parser.add_argument("--zh", action="store_true")

args = parser.parse_args()


###################################
# for win desktop probably use --server 127.0.0.1 --inbrowser
# For linux server probably use --server 127.0.0.1 or do not use any cmd flags

#return out_dotenv_loaded_models, out_dotenv_loaded_paths, out_dotenv_loaded_params

if args.checkmodels:
    lcx_checkmodels(in_dotenv_needed_models,out_dotenv_loaded_paths, out_dotenv_loaded_models, out_dotenv_loaded_models_values, in_files_to_check_in_paths )

if args.sysreport:
    full_report=generate_troubleshooting_report(in_model_config_file=in_model_config_file)
    print(full_report)
    sys.exit()

if debug_mode:
    print("---current model paths---------")
    for id in out_dotenv_loaded_models:
        print (f"{id}: {out_dotenv_loaded_models[id]}")

####################################################################################################################
####################################################################################################################
####################################################################################################################
#prefix end#########################################################################################################
#example_var=out_dotenv_loaded_params["DEBUG_MODE"]

COMPRESSION_LEVEL=out_dotenv_loaded_params["IMAGE_COMPRESSION_LEVEL_1_TO_100"]

import io

import numpy as np
import os
import torch
import random

from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights
from PIL import Image

from data.data_utils import add_special_tokens, pil_img2rgb
from data.transforms import ImageTransform
from inferencer import InterleaveInferencer
from modeling.autoencoder import load_ae
from modeling.bagel.qwen2_navit import NaiveCache
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM,
    SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer

import argparse
from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model

##LCX IMG CONVERT#######
# Dummy image generation function
def generate_image():
    img = Image.new("RGB", (200, 200), color="blue")
    return img

import tempfile

def get_image_download(img, format_choice):

    if img is None:
        return None
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    suffix = f".{format_choice.lower()}"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        save_kwargs = {}

        # Sanitize compression value
        image_compression = max(1, min(100, COMPRESSION_LEVEL))

        if format_choice.lower() == "png":
            compress_level = int((image_compression - 1) / 11)  # 1–100 -> 0–9
            save_kwargs = {"optimize": True, "compress_level": compress_level}
        elif format_choice.lower() in {"jpeg", "jpg", "webp"}:
            quality = 101 - image_compression  # 1 = 100 quality, 100 = 1 quality
            save_kwargs = {"quality": quality}

        img.save(tmp.name, format=format_choice.upper(), **save_kwargs)
        return tmp.name
####LCX IMG CONVERT END###

# Model Initialization
model_path = args.model_path #Download from https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT to models/BAGEL-7B-MoT

model_path = args.model_path

llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
llm_config.qk_norm = True
llm_config.tie_word_embeddings = False
llm_config.layer_module = "Qwen2MoTDecoderLayer"

vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
vit_config.rope = False
vit_config.num_hidden_layers -= 1

vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

config = BagelConfig(
    visual_gen=True,
    visual_und=True,
    llm_config=llm_config,
    vit_config=vit_config,
    vae_config=vae_config,
    vit_max_num_patch_per_side=70,
    connector_act='gelu_pytorch_tanh',
    latent_patch_size=2,
    max_latent_size=64,
)

with init_empty_weights():
    language_model = Qwen2ForCausalLM(llm_config)
    vit_model      = SiglipVisionModel(vit_config)
    model          = Bagel(language_model, vit_model, config)
    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

vae_transform = ImageTransform(1024, 512, 16)
vit_transform = ImageTransform(980, 224, 14)


# Model Loading and Multi GPU Infernece Preparing
device_map = infer_auto_device_map(
    model,
    max_memory=max_memory,
    no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
)

same_device_modules = [
    'language_model.model.embed_tokens',
    'time_embedder',
    'latent_pos_embed',
    'vae2llm',
    'llm2vae',
    'connector',
    'vit_pos_embed'
]

if torch.cuda.device_count() == 1:
    first_device = device_map.get(same_device_modules[0], "cuda:0")
    for k in same_device_modules:
        if k in device_map:
            device_map[k] = first_device
        else:
            device_map[k] = "cuda:0"
else:
    first_device = device_map.get(same_device_modules[0])
    for k in same_device_modules:
        if k in device_map:
            device_map[k] = first_device

if args.mode == 1:
    print("STarting in Richpants-GPU mode")
    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=os.path.join(model_path, "ema.safetensors"),
        device_map=device_map,
        offload_buffers=True,
        offload_folder=model_offload_dir,
        dtype=torch.bfloat16,
        force_hooks=True,
    ).eval()
elif args.mode == 2: # NF4
    print("STarting in consumer-GPU mode")
    bnb_quantization_config = BnbQuantizationConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=False, bnb_4bit_quant_type="nf4")
    model = load_and_quantize_model(
        model,
        weights_location=os.path.join(model_path, "ema.safetensors"),
        bnb_quantization_config=bnb_quantization_config,
        device_map=device_map,
        offload_folder=model_offload_dir,
    ).eval()
elif args.mode == 3: # INT8
    print("STarting in Potato-GPU mode")
    bnb_quantization_config = BnbQuantizationConfig(load_in_8bit=True, torch_dtype=torch.float32)
    model = load_and_quantize_model(
        model,
        weights_location=os.path.join(model_path, "ema.safetensors"),
        bnb_quantization_config=bnb_quantization_config,
        device_map=device_map,
        offload_folder=model_offload_dir,
    ).eval()
else:
    raise NotImplementedError

# Inferencer Preparing
inferencer = InterleaveInferencer(
    model=model,
    vae_model=vae_model,
    tokenizer=tokenizer,
    vae_transform=vae_transform,
    vit_transform=vit_transform,
    new_token_ids=new_token_ids,
)


def set_seed(seed):
    """Set random seeds for reproducibility"""
    if seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed

#
# # Text to Image function with thinking option and hyperparameters
# def text_to_image(prompt, show_thinking=False, cfg_text_scale=4.0, cfg_interval=0.4,
#                   timestep_shift=3.0, num_timesteps=50,
#                   cfg_renorm_min=0.0, cfg_renorm_type="global",
#                   max_think_token_n=1024, do_sample=False, text_temperature=0.3,
#                   seed=0, image_ratio="1:1"):
#     # Set seed for reproducibility
#     set_seed(seed)
#
#     if image_ratio == "1:1":
#         image_shapes = (1024, 1024)
#     elif image_ratio == "4:3":
#         image_shapes = (768, 1024)
#     elif image_ratio == "3:4":
#         image_shapes = (1024, 768)
#     elif image_ratio == "16:9":
#         image_shapes = (576, 1024)
#     elif image_ratio == "9:16":
#         image_shapes = (1024, 576)
#
#         # Set hyperparameters
#     inference_hyper = dict(
#         max_think_token_n=max_think_token_n if show_thinking else 1024,
#         do_sample=do_sample if show_thinking else False,
#         text_temperature=text_temperature if show_thinking else 0.3,
#         cfg_text_scale=cfg_text_scale,
#         cfg_interval=[cfg_interval, 1.0],  # End fixed at 1.0
#         timestep_shift=timestep_shift,
#         num_timesteps=num_timesteps,
#         cfg_renorm_min=cfg_renorm_min,
#         cfg_renorm_type=cfg_renorm_type,
#         image_shapes=image_shapes,
#     )
#
#     # Call inferencer with or without think parameter based on user choice
#     result = inferencer(text=prompt, think=show_thinking, **inference_hyper)
#     return result["image"], result.get("text", None)
#

# Image Understanding function with thinking option and hyperparameters
def image_understanding(image: Image.Image, prompt: str, show_thinking=False,
                        do_sample=False, text_temperature=0.3, max_new_tokens=512):
    if image is None:
        return "Please upload an image."

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    image = pil_img2rgb(image)

    # Set hyperparameters
    inference_hyper = dict(
        do_sample=do_sample,
        text_temperature=text_temperature,
        max_think_token_n=max_new_tokens, # Set max_length
    )

    # Use show_thinking parameter to control thinking process
    result = inferencer(image=image, text=prompt, think=show_thinking,
                        understanding_output=True, **inference_hyper)
    return result["text"]

#
# # Image Editing function with thinking option and hyperparameters
# def edit_image(image: Image.Image, prompt: str, show_thinking=False, cfg_text_scale=4.0,
#                cfg_img_scale=2.0, cfg_interval=0.0,
#                timestep_shift=3.0, num_timesteps=50, cfg_renorm_min=0.0,
#                cfg_renorm_type="text_channel", max_think_token_n=1024,
#                do_sample=False, text_temperature=0.3, seed=0):
#     # Set seed for reproducibility
#     set_seed(seed)
#
#     if image is None:
#         return "Please upload an image.", ""
#
#     if isinstance(image, np.ndarray):
#         image = Image.fromarray(image)
#
#     image = pil_img2rgb(image)
#
#     # Set hyperparameters
#     inference_hyper = dict(
#         max_think_token_n=max_think_token_n if show_thinking else 1024,
#         do_sample=do_sample if show_thinking else False,
#         text_temperature=text_temperature if show_thinking else 0.3,
#         cfg_text_scale=cfg_text_scale,
#         cfg_img_scale=cfg_img_scale,
#         cfg_interval=[cfg_interval, 1.0],  # End fixed at 1.0
#         timestep_shift=timestep_shift,
#         num_timesteps=num_timesteps,
#         cfg_renorm_min=cfg_renorm_min,
#         cfg_renorm_type=cfg_renorm_type,
#     )
#
#     # Include thinking parameter based on user choice
#     result = inferencer(image=image, text=prompt, think=show_thinking, **inference_hyper)
#     return result["image"], result.get("text", "")


# Helper function to load example images
def load_example_image(image_path):
    try:
        return Image.open(image_path)
    except Exception as e:
        print(f"Error loading example image: {e}")
        return None


if __name__ == "__main__":
    result = image_understanding(
        load_example_image('1.jpeg'), "判断画面中是否有烟火", False,
        False, 0.3, 512
    )
    print(result)