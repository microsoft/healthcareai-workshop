# Environment Setup Instructions

This document outlines how to set up your environment with Azure ML SDK v2.

**Step 1: Install Required Tools**  
- Install [Python 3.x](https://www.python.org) and create a virtual environment.  
- Install VSCode.  
- Install the Azure CLI v2: `az upgrade`  
- Install the Azure ML SDK v2:  
  ```
  pip install azure-ai-ml azure-identity
  ```

**Step 2: Configure Authentication**  
- Use `DefaultAzureCredential` for interactive login or service principal authentication.  
- Set any required environment variables.

**Step 3: Set Up the Azure ML Workspace**  
- Use the provided configuration (e.g., `azureml_config.json` placed in a config folder) for workspace connection.  
- Verify connectivity by running the test script `environment_test.py`.

**Step 4: (Optional) Configure CI/CD**  
- Integrate with GitHub Actions or Azure DevOps for automated environment provisioning.