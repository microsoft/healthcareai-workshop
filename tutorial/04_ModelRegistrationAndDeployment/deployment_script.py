from azure.ai.ml import MLClient
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment, CodeConfiguration
from azure.identity import DefaultAzureCredential

def deploy_model(ml_client):
    # Step 1: Create (or update) an online endpoint
    endpoint = ManagedOnlineEndpoint(
        name="my-model-endpoint",
        description="Endpoint for model inference"
    )
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    print("Endpoint created or updated:", endpoint.name)
    
    # Step 2: Create (or update) an online deployment
    deployment = ManagedOnlineDeployment(
        name="blue",
        endpoint_name=endpoint.name,
        model=ml_client.models.get("my_model", label="latest"),
        code_configuration=CodeConfiguration(
            code=".",  # Folder containing your scoring.py file
            scoring_script="scoring.py"
        ),
        environment="your_environment_name",
        instance_type="Standard_DS3_v2",
        instance_count=1
    )
    ml_client.online_deployments.begin_create_or_update(deployment).result()
    print("Deployment completed. Endpoint is live.")

if __name__ == "__main__":
    credential = DefaultAzureCredential()
    ml_client = MLClient.from_config(credential=credential)
    deploy_model(ml_client)