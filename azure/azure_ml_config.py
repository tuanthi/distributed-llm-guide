"""
Azure ML configuration and deployment utilities for Distributed LLM models.
Supports distributed training, model registration, and endpoint deployment.
"""

import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path

from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    Environment,
    Model,
    CodeConfiguration,
    Command,
    Compute,
    AmlCompute,
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    OnlineRequestSettings,
    ProbeSettings,
    ResourceRequirementsSettings,
    ResourceSettings,
)
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential
from azure.ai.ml.sweep import Choice, Uniform, BayesianSamplingAlgorithm
from loguru import logger


@dataclass
class AzureMLConfig:
    """Configuration for Azure ML workspace."""
    
    subscription_id: str
    resource_group: str
    workspace_name: str
    
    # Compute configuration
    compute_name: str = "gpu-cluster"
    compute_size: str = "Standard_NC24ads_A100_v4"
    min_instances: int = 0
    max_instances: int = 4
    
    # Environment
    environment_name: str = "distributed-llm-training-env"
    environment_version: str = "1.0"
    
    # Model registration
    model_name: str = "distributed-llm-model"
    model_version: str = "1.0"
    
    # Endpoint configuration
    endpoint_name: str = "distributed-llm-endpoint"
    deployment_name: str = "distributed-llm-deployment"
    instance_type: str = "Standard_DS3_v2"
    instance_count: int = 1


class AzureMLManager:
    """Manages Azure ML operations for model training and deployment."""
    
    def __init__(self, config: AzureMLConfig):
        self.config = config
        self.ml_client = self._create_ml_client()
        
    def _create_ml_client(self) -> MLClient:
        """Create Azure ML client."""
        credential = DefaultAzureCredential()
        
        ml_client = MLClient(
            credential=credential,
            subscription_id=self.config.subscription_id,
            resource_group_name=self.config.resource_group,
            workspace_name=self.config.workspace_name,
        )
        
        logger.info(f"Connected to Azure ML workspace: {self.config.workspace_name}")
        return ml_client
        
    def create_compute_cluster(self) -> AmlCompute:
        """Create or update compute cluster for training."""
        try:
            compute = self.ml_client.compute.get(self.config.compute_name)
            logger.info(f"Compute cluster {self.config.compute_name} already exists")
            return compute
        except:
            logger.info(f"Creating compute cluster: {self.config.compute_name}")
            
        compute_config = AmlCompute(
            name=self.config.compute_name,
            type="amlcompute",
            size=self.config.compute_size,
            min_instances=self.config.min_instances,
            max_instances=self.config.max_instances,
            idle_time_before_scale_down=120,
            tier="Dedicated",
        )
        
        compute = self.ml_client.compute.begin_create_or_update(compute_config).result()
        logger.info(f"Compute cluster {self.config.compute_name} created successfully")
        
        return compute
        
    def create_environment(self) -> Environment:
        """Create custom environment for training."""
        try:
            env = self.ml_client.environments.get(
                self.config.environment_name,
                version=self.config.environment_version
            )
            logger.info(f"Environment {self.config.environment_name} already exists")
            return env
        except:
            logger.info(f"Creating environment: {self.config.environment_name}")
            
        # Docker image with CUDA and ML frameworks
        docker_image = "mcr.microsoft.com/azureml/curated/pytorch-2.0-cuda11.7-py38-ubuntu20.04:latest"
        
        conda_dependencies = """
channels:
  - conda-forge
  - pytorch
  - nvidia
dependencies:
  - python=3.8
  - pytorch::pytorch>=2.0.0
  - pytorch::torchvision
  - pytorch::torchaudio
  - pytorch::pytorch-cuda=11.7
  - transformers>=4.36.0
  - datasets>=2.14.0
  - accelerate>=0.25.0
  - deepspeed>=0.12.0
  - peft>=0.7.0
  - bitsandbytes>=0.41.0
  - wandb
  - mlflow
  - loguru
  - rich
  - typer
  - pip
  - pip:
    - azure-ai-ml
    - azure-storage-blob
    - azureml-core
    - onnx
    - onnxruntime-gpu
    - triton-python-bindings
    - einops
    - safetensors
"""
        
        env = Environment(
            name=self.config.environment_name,
            version=self.config.environment_version,
            image=docker_image,
            conda_file="environment.yml",
            description="Distributed LLM training environment with GPU support",
        )
        
        # Save conda file
        conda_path = Path("environment.yml")
        conda_path.write_text(conda_dependencies)
        
        env = self.ml_client.environments.create_or_update(env)
        logger.info(f"Environment {self.config.environment_name} created successfully")
        
        return env
        
    def submit_training_job(
        self,
        script_path: str,
        experiment_name: str,
        parameters: Optional[Dict[str, Any]] = None,
        node_count: int = 1,
    ) -> Any:
        """Submit distributed training job to Azure ML."""
        
        # Ensure compute and environment exist
        self.create_compute_cluster()
        self.create_environment()
        
        # Default parameters
        if parameters is None:
            parameters = {
                "model_name": "microsoft/phi-2",
                "dataset_name": "codeparrot/github-code",
                "output_dir": "./outputs",
                "learning_rate": 5e-5,
                "num_train_epochs": 3,
                "per_device_train_batch_size": 4,
                "gradient_accumulation_steps": 8,
            }
            
        # Create job configuration
        job_config = Command(
            code="./src",
            command=f"python {script_path} " + " ".join([
                f"--{key} {value}" for key, value in parameters.items()
            ]),
            environment=f"{self.config.environment_name}:{self.config.environment_version}",
            compute=self.config.compute_name,
            instance_count=node_count,
            distribution={
                "type": "PyTorch",
                "process_count_per_instance": 1,
            } if node_count > 1 else None,
            display_name=f"Distributed LLM Training - {experiment_name}",
            description="Distributed training job for Distributed LLM models",
            tags={"framework": "pytorch", "task": "training"},
        )
        
        # Submit job
        job = self.ml_client.jobs.create_or_update(job_config)
        logger.info(f"Training job submitted: {job.name}")
        
        return job
        
    def register_model(
        self,
        model_path: str,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> Model:
        """Register trained model in Azure ML."""
        
        model_name = model_name or self.config.model_name
        model_version = model_version or self.config.model_version
        
        model = Model(
            path=model_path,
            name=model_name,
            version=model_version,
            description=description or f"Distributed LLM model version {model_version}",
            tags=tags or {"framework": "pytorch", "task": "language_modeling"},
            type=AssetTypes.CUSTOM_MODEL,
        )
        
        registered_model = self.ml_client.models.create_or_update(model)
        logger.info(f"Model registered: {registered_model.name}:{registered_model.version}")
        
        return registered_model
        
    def create_online_endpoint(
        self,
        endpoint_name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> ManagedOnlineEndpoint:
        """Create managed online endpoint for model serving."""
        
        endpoint_name = endpoint_name or self.config.endpoint_name
        
        try:
            endpoint = self.ml_client.online_endpoints.get(endpoint_name)
            logger.info(f"Endpoint {endpoint_name} already exists")
            return endpoint
        except:
            logger.info(f"Creating endpoint: {endpoint_name}")
            
        endpoint = ManagedOnlineEndpoint(
            name=endpoint_name,
            description=description or "Distributed LLM model serving endpoint",
            auth_mode="key",
            tags={"framework": "pytorch", "task": "inference"},
        )
        
        endpoint = self.ml_client.online_endpoints.begin_create_or_update(endpoint).result()
        logger.info(f"Endpoint {endpoint_name} created successfully")
        
        return endpoint
        
    def deploy_model(
        self,
        model_name: str,
        model_version: str,
        scoring_script: str,
        endpoint_name: Optional[str] = None,
        deployment_name: Optional[str] = None,
        instance_type: Optional[str] = None,
        instance_count: Optional[int] = None,
    ) -> ManagedOnlineDeployment:
        """Deploy model to online endpoint."""
        
        endpoint_name = endpoint_name or self.config.endpoint_name
        deployment_name = deployment_name or self.config.deployment_name
        instance_type = instance_type or self.config.instance_type
        instance_count = instance_count or self.config.instance_count
        
        # Ensure endpoint exists
        self.create_online_endpoint(endpoint_name)
        
        # Get registered model
        model = self.ml_client.models.get(model_name, version=model_version)
        
        deployment = ManagedOnlineDeployment(
            name=deployment_name,
            endpoint_name=endpoint_name,
            model=model,
            code_configuration=CodeConfiguration(
                code="./deployment",
                scoring_script=scoring_script,
            ),
            environment=f"{self.config.environment_name}:{self.config.environment_version}",
            instance_type=instance_type,
            instance_count=instance_count,
            request_settings=OnlineRequestSettings(
                request_timeout_ms=60000,
                max_concurrent_requests_per_instance=4,
                max_queue_wait_ms=120000,
            ),
            liveness_probe=ProbeSettings(
                failure_threshold=30,
                success_threshold=1,
                timeout=2,
                period=10,
                initial_delay=10,
            ),
            readiness_probe=ProbeSettings(
                failure_threshold=30,
                success_threshold=1,
                timeout=2,
                period=10,
                initial_delay=10,
            ),
            resource_requirements=ResourceRequirementsSettings(
                cpu="2",
                memory="8Gi",
            ),
        )
        
        deployment = self.ml_client.online_deployments.begin_create_or_update(deployment).result()
        
        # Set traffic to 100% for this deployment
        endpoint = self.ml_client.online_endpoints.get(endpoint_name)
        endpoint.traffic = {deployment_name: 100}
        self.ml_client.online_endpoints.begin_create_or_update(endpoint).result()
        
        logger.info(f"Model deployed to endpoint {endpoint_name}/{deployment_name}")
        
        return deployment
        
    def create_hyperparameter_sweep(
        self,
        script_path: str,
        experiment_name: str,
        parameter_space: Dict[str, Any],
        max_total_trials: int = 20,
        max_concurrent_trials: int = 4,
    ) -> Any:
        """Create hyperparameter sweep job."""
        
        from azure.ai.ml.sweep import SweepJob
        
        # Base command
        base_job = Command(
            code="./src",
            command=f"python {script_path}",
            environment=f"{self.config.environment_name}:{self.config.environment_version}",
            compute=self.config.compute_name,
            instance_count=1,
        )
        
        # Create sweep job
        sweep_job = SweepJob(
            trial=base_job,
            sampling_algorithm=BayesianSamplingAlgorithm(),
            primary_metric="eval_loss",
            goal="minimize",
            max_total_trials=max_total_trials,
            max_concurrent_trials=max_concurrent_trials,
            search_space=parameter_space,
            display_name=f"Distributed LLM Hyperparameter Sweep - {experiment_name}",
            description="Hyperparameter optimization for Distributed LLM models",
        )
        
        # Submit sweep
        sweep_job = self.ml_client.jobs.create_or_update(sweep_job)
        logger.info(f"Hyperparameter sweep submitted: {sweep_job.name}")
        
        return sweep_job
        
    def monitor_job(self, job_name: str) -> Dict[str, Any]:
        """Monitor job status and metrics."""
        job = self.ml_client.jobs.get(job_name)
        
        return {
            "name": job.name,
            "status": job.status,
            "creation_time": job.creation_context.created_at,
            "compute": job.compute,
            "tags": job.tags,
        }
        
    def download_model(self, model_name: str, model_version: str, download_path: str):
        """Download registered model to local path."""
        self.ml_client.models.download(
            name=model_name,
            version=model_version,
            download_path=download_path
        )
        
        logger.info(f"Model {model_name}:{model_version} downloaded to {download_path}")


def create_default_config() -> AzureMLConfig:
    """Create default Azure ML configuration from environment variables."""
    return AzureMLConfig(
        subscription_id=os.getenv("AZURE_SUBSCRIPTION_ID", ""),
        resource_group=os.getenv("AZURE_RESOURCE_GROUP", "distributed-llm-rg"),
        workspace_name=os.getenv("AZURE_ML_WORKSPACE", "distributed-llm-workspace"),
        compute_name=os.getenv("AZURE_COMPUTE_NAME", "gpu-cluster"),
        environment_name=os.getenv("AZURE_ENVIRONMENT_NAME", "distributed-llm-env"),
        model_name=os.getenv("AZURE_MODEL_NAME", "distributed-llm-model"),
        endpoint_name=os.getenv("AZURE_ENDPOINT_NAME", "distributed-llm-endpoint"),
    )