# What is *act* ?

When creating a workflow for GitHub Actions, one has to create a YAML-file, commit it and push it to GitHub. Then only on GitHub one can test wether there is an error in the workflow or not. So debugging the workflow can be quite enoying and time consuming. This is solved by *act*, a tool for running your GitHub Actions locally using the Docker API.

## Installation guide for Windows

For testing wether everything has been installed well, we recommend to clone the github demo project by running `git clone https://github.com/cplee/github-actions-demo.git` in a directory of your choice.

###### Install Docker

* Download the .exe file (Docker Desktop for Windows - x86_64) from [https://docs.docker.com/desktop/setup/install/windows-install/](https://docs.docker.com/desktop/setup/install/windows-install/)
* Execute it and register for docker.
* Run docker EACH TIME you want to use *act*

###### Install act

* Use winget to install act: `winget install nektos.act`
* Restart your terminal
* Change your working directory to the demo project: `cd YOUR_PATH/github-actions-demo/`
* List all workflows using *`act --list`*
* Now run the job called 'test' using `act -j test`. Because act is running for the first time this will take a while and you will be asked: `Please choose the default image you want to use …` Choose between Medium or Large and proceed. Micro is also a valid option but it will lead to the known issue [Unable to execute actions/setup-python](https://github.com/nektos/act/issues/251) when using the common action setup-python.
* Enjoy the displayed   **`🏁 Job succeeded`**

## Common flags

* -n ,  --dryrun	Validates workflow correctness without container creation
* -g ,  --graph	Draw the workflow
* -l ,  --list		List all workflows (Stage  Job ID  Job name  Workflow name  Workflow file  Events)
* -j ,  --job ID	Run a specific job using its ID
