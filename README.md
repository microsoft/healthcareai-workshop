# Medical X-ray Pneumonia Detection Workshop

This repository contains a comprehensive end-to-end machine learning workflow for developing, training, registering, and deploying a pneumonia detection model using chest X-ray images. The tutorial leverages Azure Machine Learning for scalable training, model versioning, and deployment.

## Prerequisites

Before beginning this tutorial, please ensure you have the following:

- An Azure subscription with sufficient permissions to create resources
- Azure Machine Learning workspace provisioned
- Visual Studio Code (VS Code) installed on your local machine
- Basic familiarity with Python, PyTorch, and machine learning concepts
- Required Python packages installed

### Installing Visual Studio Code and Required Extensions

To set up Visual Studio Code (VS Code) for this workshop, follow these steps:

1. **Download and Install VS Code**  
   - Visit the [Visual Studio Code website](https://code.visualstudio.com/).
   - Download the appropriate installer for your operating system (Windows, macOS, or Linux).
   - Run the installer and follow the on-screen instructions to complete the installation.

2. **Install Required Extensions**  
   Once VS Code is installed, open it and install the following extensions to enhance your development experience:

      - **Azure Machine Learning**  
      - **Azure ML - Remote**  
      - **Azure Resources**  
      - **Jupyter**  
      - **Pylance**  
      - **Python**  
      - **Python Debugger**  
      - **Remote - SSH**  

3. **Verify Extensions**  
   - Open the Extensions view in VS Code by clicking on the Extensions icon in the Activity Bar on the side of the window or pressing `Ctrl+Shift+X` (Windows/Linux) or `Cmd+Shift+X` (macOS).
   - Search for each extension by name, ensure it is marked as **@installed**, and click **Install** if it is not already installed.

Once VS Code and the required extensions are installed, you are ready to proceed with the workshop.  Here is the offical documentation for setting up vscode: [docs](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-setup-vs-code?view=azureml-api-2).

## Environment Setup

### 1. Create a CPU Compute Instance

To create a CPU compute instance, follow these steps:

1. Log in to the [Azure Machine Learning Studio](https://ml.azure.com/).
2. Navigate to your Azure ML workspace.
3. Select "Compute" from the left-hand menu.
4. Under the "Compute Instances" tab, click on the **+ New** button.
5. Fill in the required details:
    - **Virtual Machine Size**: Choose a CPU-based VM (e.g., `Standard_DS11_v2`).
    - **Compute Name**: Provide a unique name for your compute instance.
6. Click **Create** to provision the compute instance.
7. Wait for the compute instance to transition to the "Running" state before proceeding.
8. Configure the compute instance to stay on for 5 hours:
    - Select the compute instance from the list.
    - Click on the **Schedule** tab.
    - Add a new schedule to keep the instance running for 5 hours.
    - Save the schedule configuration.

Once the compute instance is ready, you can use it for running the notebooks and other code.

### 2. Connect to Your Compute with VSCode

1. **Log in to Azure Portal**  
   Open [AzureML Studdio](https://ml.azure.com) and log in with your credentials.

2. **Navigate to Azure Machine Learning Workspace**  
   - In the search bar at the top, type the name of the workspace and select it from the results.
   - Open your Azure Machine Learning workspace.

3. **Go to the Compute Section**  
   - In the left-hand menu, click on **Compute** under the **Manage** section.

4. **Select the Compute Instance**  
   - In the **Compute Instances** tab, locate the compute instance you want to connect to (e.g., `jmerkow1`).
   - Ensure the compute instance is in the **Running** state. If it is not running, click **Start** to activate it.

5. **Open VS Code (Desktop)**  
   - Click the **...** (ellipsis) next to the compute instance name.
   - From the dropdown menu, select **VS Code (Desktop)**.

6. **Follow the Connection Instructions**  
   - If prompted, follow the instructions to open VS Code on your local machine and connect to the remote compute instance.
   - The vscode server will need a few moments to initialize the first time you connect from vscode.

Once connected, you can access the files and run the notebooks directly from the compute instance using VS Code.

> Note:  You can also run the demo locally, you will still need an AzureML workspace for data, and training resources.  Follow instructions [here](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-configure-cli?view=azureml-api-2).


## Setup Remote Environment

To set up the required environment, follow these steps:

1. **Open the Terminal in VS Code**
Press `Ctrl+` (backtick) or go to View → Terminal from the menu bar to open the integrated terminal.

2. **Clone the Repository**  
   Begin by cloning this repository to your remote compute instance:

   ```bash
   cd ~/cloudfiles/code/Users/<alias>/ && git clone https://github.com/microsoft/healtcareai-workshop
   cd healtcareai-workshop
   ```

3. **Set Up Conda Environment**  
   Before installing dependencies, ensure you are using the correct Conda environment for this workshop. Follow these steps:

   - Switch to the `azureml_py310_sdkv2` Conda environment:
     ```bash
     conda activate azureml_py310_sdkv2
     ```

   - You may encounter a known issue when activating the environment, if you do, you need to deactivate multiple times before switching:
     ```bash
     conda deactivate
     conda deactivate
     conda activate azureml-py310-sdkv2
     ```
   - Ensure you are not already in another Conda environment before activating `azureml_py310_sdkv2`.

   This will be the environment you use for all workshop activities.

4. **Install Dependencies**  
   Use the provided `requirements.txt` file located in the root directory of this repository to install the necessary Python packages:

   ```bash
   pip install -r requirements.txt
   ```

5. **Complete Environment Setup**  
   Execute the following commands to finalize the environment setup:

   ```bash
   conda install -y conda-build
   conda develop src
   ```

Once these steps are completed, your environment will be ready for running the workshop notebooks and scripts.


## Workshop Structure

TYou are now ready to start the tutorial, In each notebook, make sure to select the his tutorial follows a systematic approach to ML model development:

1. **Environment Setup** (`01_EnvironmentSetup`)
   - Configure Azure ML workspace.
   - Set up compute resources.
   - Validate environment connectivity.

2. **Data Preparation** (`02_DataPreparation`)
   - Download and explore the dataset.
   - Split data into training and validation sets.
   - Upload and register dataset in AzureML.

3. **Model Training** (`03_Training`)
   - Define model architecture using PyTorch Lightning
   - Configure hyperparameter tuning
   - Submit distributed training jobs to AzureML

4. **Model Registration and Deployment** (`04_ModelRegistrationAndDeployment`)
   - Register the best model version
   - Deploy model to a real-time inference endpoint
   - Test the deployed model with real X-ray images

## Getting Started

1. Complete all prerequisites 
2. Run through the notebooks in sequential order:
   - `tutorial/01_EnvironmentSetup/environment_check.ipynb`
   - `tutorial/02_DataPreparation/data_preparation.ipynb`
   - `tutorial/03_Training/train-job.ipynb`
   - `tutorial/04_ModelRegistrationAndDeployment/model_reg_and_deploy.ipynb`

Each notebook contains detailed instructions and explanations for each step of the process.
