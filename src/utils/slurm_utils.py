import subprocess
import shlex
import os
import logging

from typing import List


def run_process(cmd_line: str) -> None:
    """Convenience method to wrap a process and log the output
    Parameters
    ----------
    cmd_line (str): 
        command line to be executed
    """

    cmd = shlex.split(cmd_line)
    logging.debug(cmd)
    try:
        process_output = subprocess.run(
            cmd, universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        logging.info(process_output)
        return process_output
    except subprocess.CalledProcessError as e:
        logging.error("Retcode:%s, CMD:%s, Output:%s" % (e.returncode, e.cmd, e.output))
    except os.OSError as e:
        logging.error(e)


def batch_job(
    command: str,
    array_start: int = 0,
    array_end: int = 0,
    modules: List[str] = None,
    max_parallel: int = 0,
    conda_env: str = None,
    cores: int = 0,
    mem: str = None,
    wait: bool = False,
    output: str = None,
    requested_time: str = None,
    partition: str = None,
    reservation: str = None,
    gres: str = None,
    chdir: str = None,
):
    """Create a job array
    Parameters
    ----------
    command: str
        Command to run in the array
    array_start: int, optional
        Index of the first job (usually 0 or 1)
    array_end: int, optional
        Index of the last job
    modules: List[str], optional
        List of env. modules to load to run the job ["anaconda3", "hdf5/1.10.0", ...]
    max_parallel: int, optional
        The maximum number of jobs to run in parallel
    conda_env: str, optional
        If we need to run this in a comnda environment, specify which one
    cores: int, 0
        How many cores per job, default = 0 = system default
    mem: str, optional
        How much memory per job, specify as <num>G/M, f=default = 8G
    wait: bool, optional
        Execute the command in the foreground, default = True
    output: str, optional
        Output pattern where to write the slurm logs, default to current directory with default slurm naming
    requested_time: str, optional
        How much time we request per job. This is important for shorter jobs, so that thay can be faster scheduled 
        default = None. meaning whatever is the default for the current partition/QoS
    partition: str, optional
        What Slurm partition do we want to use, defaut = none (=up-cpu)
    reservation: str, optional
        What Slurm reservation do we want to run this (e.g. ai_ml)
    gres: str, optional
        What Slurm GRES do we want to use, for example with GPUs = e.g. gpu:V100:2, default = None
    chdir: str
        Change to this directory as the base for the job
    """
    slurm_options = {}

    # Put together array options if any
    is_array = array_start >= 0 and array_end > 0 and array_end > array_start
    if is_array:
        slurm_options["array"] = f"--array={array_start}-{array_end}"
        if max_parallel > 0:
            slurm_options["array"] = f"{slurm_options['array']}%{max_parallel}"

    if cores > 0:
        slurm_options["cores"] = f"--cpus-per-task={cores}"

    if mem:
        slurm_options["mem"] = f"--mem={mem}"

    if wait:
        slurm_options["wait"] = "--wait"

    if output:
        slurm_options["output"] = f"--output={output}"

    if partition:
        slurm_options["partition"] = f"--partition={partition}"

    if reservation:
        slurm_options["reservation"] = f"--reservation={reservation}"

    if gres:
        slurm_options["gres"] = f"--gres={gres}"

    if requested_time:
        slurm_options["time"] = f"--time={requested_time}"

    if chdir:
        slurm_options["chdir"] = f"--chdir={chdir}"

    module_cmd = ""
    if modules:
        module_cmd = " module load {} && ".format(" ".join(modules))

    conda_cmd = ""
    if conda_env:
        conda_cmd = f" source activate {conda_env} && "

    slurm_cmd_str = 'sbatch {} --wrap="{} {} {}"'.format(
        " ".join(list(slurm_options.values())), module_cmd, conda_cmd, command
    )
    logging.info(f"Running: {slurm_cmd_str}")

    job_id = run_process(slurm_cmd_str)
    job_id = job_id.stdout.strip().split(" ")[-1]
    return job_id
