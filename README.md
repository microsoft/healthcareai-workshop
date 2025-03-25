# Medical X-ray Pneumonia Detection Workshop

This repository contains a comprehensive end-to-end machine learning workflow for developing, training, registering, and deploying a pneumonia detection model using chest X-ray images. The tutorial leverages Azure Machine Learning for scalable training, model versioning, and deployment.

## Prerequisites

Before beginning this tutorial, please ensure you have the following:

- An Azure subscription with sufficient permissions to create resources
- Azure Machine Learning workspace provisioned
- Basic familiarity with Python, PyTorch, and machine learning concepts
- Required Python packages installed:

To set up the required environment, ensure you have the following dependencies installed.  Use the provided `requirements.txt` file located in the root directory of this repository:

```bash
pip install -r requirements.txt
```

Additionally, execute the following commands to complete the environment setup:

```bash
conda install conda-build
conda develop src
```

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
   Open [Azure Portal](https://portal.azure.com) and log in with your credentials.

2. **Navigate to Azure Machine Learning Workspace**  
   - In the search bar at the top, type **"Machine Learning"** and select **Machine Learning** from the results.
   - Open your Azure Machine Learning workspace.

3. **Go to the Compute Section**  
   - In the left-hand menu, click on **Compute** under the **Manage** section.

4. **Select the Compute Instance**  
   - In the **Compute Instances** tab, locate the compute instance you want to connect to (e.g., `jmerkow-cpu-uw`).
   - Ensure the compute instance is in the **Running** state. If it is not running, click **Start** to activate it.

5. **Open VS Code (Desktop)**  
   - Click the **...** (ellipsis) next to the compute instance name.
   - From the dropdown menu, select **VS Code (Desktop)**.

6. **Follow the Connection Instructions**  
   - If prompted, follow the instructions to open VS Code on your local machine and connect to the remote compute instance.

Once connected, you can access the files and run the notebooks directly from the compute instance using VS Code.
## Workshop Structure

This tutorial follows a systematic approach to ML model development:

1. **Environment Setup** (`01_EnvironmentSetup`)
   - Configure Azure ML workspace
   - Set up compute resources
   - Validate environment connectivity

2. **Data Preparation** (`02_DataPreparation`)
   - Download and explore the RSNA Pneumonia Detection dataset
   - Preprocess X-ray images
   - Split data into training and validation sets

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
   - environment_check.ipynb
   - data_preparation.ipynb 
   - train-job.ipynb
   - model_reg_and_deploy.ipynb

Each notebook contains detailed instructions and explanations for each step of the process.