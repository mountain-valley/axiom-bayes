# 1 Getting Acquainted

## 1.1 Supercomputer

### 1.1.1 Requesting an Account

1. Request an account if you do not already have one: [https://rc.byu.edu/](https://rc.byu.edu/)
2. Please use Dr. Wingate as your sponsor. His username is  `dw87` if needed.

### 1.1.2 Login Nodes

1. The supercomputer contains login and compute nodes. Login nodes have internet access and are used to prepare compute jobs for submission. You can think of login nodes as being CPU machines with internet access. This is also where you will submit jobs using Slurm.
2. You can login to the supercomputer by following the instructions here: [https://rc.byu.edu/wiki/?id=Logging+In](https://rc.byu.edu/wiki/?id=Logging+In) or `ssh username@ssh.rc.byu.edu`.
3. Two-factor authentication is required; however SSH multiplexing will make it easier: [https://rc.byu.edu/wiki/index.php?page=SSH+Multiplexing](https://rc.byu.edu/wiki/index.php?page=SSH+Multiplexing).
4. VS Code can be used by installing the ‘Remote Development’ pack by Microsoft. Going to ‘Remote Explorer side tab’ > SSH > Settings and adding:
    
    ```bash
    Host orc
        User username
        HostName ssh.rc.byu.edu
    ```
    
    To login, you can simply use `ssh orc` .
    
5. MacOS users can use SSH multiplexing to avoid having to do two-factor authentication every time by adding:
    
    ```bash
    Host orc
        User username
        HostName ssh.rc.byu.edu
        ControlMaster auto
        ControlPath ~/.ssh/master-%r@%h:%p.socket
        ControlPersist yes 
        ForwardX11 yes
        ServerAliveInterval 300
        XAuthLocation /opt/X11/bin/xauth
    ```
    
6. See this link to learn more about storage on the supercomputer: [https://rc.byu.edu/wiki/?id=Storage](https://rc.byu.edu/wiki/?id=Storage).

### 1.1.3 Compute Nodes

1. Compute nodes do not have access to the internet which means that data and models need to be downloaded beforehand and logging needs to be done offline.
2. Compute nodes can be used in either two ways: interactively or through jobs. Submitting jobs is preferred as letting GPUs sit idle while developing is inefficient. For reference, an A100 GPU costs ~$1.29 on [https://lambda.ai/pricing#on-demand](https://lambda.ai/pricing#on-demand).
3. Compute nodes can be used interactively by salloc-ing a node with:
    
    ```bash
    salloc --time=4:00:00 --qos=dw87 --gpus=1 --mem=32G --cpus-per-gpu=8
    ```
    
4. Jobs can be submitted using `sbatch` (preferred):
    
    ```bash
    sbatch script.sh
    ```
    
    In this example, arguments can be added in the command line or in the top of the .sh file like:
    
    ```bash
    #SBATCH --job-name=simclr_one
    #SBATCH --output=slurm_logs/%x_%j.out
    #SBATCH --gres=gpu:a100:1
    #SBATCH --cpus-per-task=8
    #SBATCH --mem=32G
    #SBATCH --time=04:00:00
    #SBATCH --qos=dw87
    ```
    
5. A list of jobs can be found using `squeue` which can be useful when paired with `squeue | grep username` or `squeue -u username`.
6. Jobs can be canceled using `scancel JOBID` or `scancel -u username` to cancel all your jobs.

## 1.2 Recommendation for Deep Learning Workflows

1. The general workflow for deep learning can be summarized by starting small and slowly scaling up while being able to iterate quickly.
2. For me, this means starting with a Jupyter Notebook on a CPU or small GPU. This is where I figure out the data, model, and evaluation. I’ll also start with a small model and data.
3. Once I’m at a good spot, I will transition to compute nodes by submitting small jobs and debugging issues that arise such as environment inconsistencies or CUDA errors. If the logs aren’t enough or if I’m consistently running into problems, I will ssh into the compute node of a running job or interact with a compute node directly. Sometimes, job queues can back up which might require salloc-ing a node to debug final errors so that when your job eventually gets executed, it will be successful.   
4. The last recommendation is to begin with the entire workflow in mind which means thinking about the data, model, training, evaluation, loss, and logging throughout.
