import sagemaker
from sagemaker.sklearn.estimator import SKLearn
from sagemaker import get_execution_role
import boto3
import os
import datetime

# --- Configuration ---

# 1. IAM Role Name
# As you've created 'IAM-Sagemaker-execution-Role'.
# If running outside a SageMaker Notebook, we will fetch its ARN.
role_name = "IAM-Sagemaker-execution-Role"

# 2. S3 Bucket for Artifacts
# Make sure your IAM role has permissions for this bucket.
bucket_name = "rrr-mlops-sagemaker"
prefix = "vehicle-insurance-prediction" # A folder within the bucket

# 3. AWS Region
# The region where your bucket and SageMaker resources are located.
region_name = "us-west-2"

# 4. MongoDB Connection URL
# IMPORTANT: Replace with your actual MongoDB connection string.
MONGODB_URL = "mongodb+srv://ravipanavirs_db_user:Tw2nC1vnfh5gY5u2@cluster0.rgjg5wy.mongodb.net/?appName=Cluster0"

# --- SageMaker Execution ---

# 1. Set up SageMaker Session and IAM Role
# Create a boto3 session with the specified region
boto_session = boto3.Session(region_name=region_name)

# Create a sagemaker session from the boto3 session
sagemaker_session = sagemaker.Session(boto_session=boto_session, default_bucket=bucket_name)

try:
    # If running in SageMaker (e.g., a notebook), it gets the role automatically.
    role_arn = get_execution_role()
    print(f"Running in SageMaker environment. Using role: {role_arn}")
except Exception:
    # If running locally, construct the role ARN from the name provided.
    print("Running in a local environment. Fetching ARN for role.")
    try:
        iam_client = boto_session.client("iam")
        role_arn = iam_client.get_role(RoleName=role_name)["Role"]["Arn"]
        print(f"Successfully fetched ARN for role '{role_name}': {role_arn}")
    except Exception as e:
        if "SignatureDoesNotMatch" in str(e) or "Signature expired" in str(e):
            print("\n" + "!"*60)
            print("CRITICAL ERROR: System Clock Out of Sync")
            print(f"AWS rejected the request because your system clock is incorrect.")
            print(f"Your Local Time (UTC): {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")
            print("Please sync your Windows Date & Time settings immediately.")
            print("!"*60 + "\n")
        print(f"Error fetching IAM role ARN. Please ensure the role '{role_name}' exists and your AWS credentials are configured.")
        raise e

# 2. Create the SKLearn Estimator
# This defines the training job configuration.
print("\nConfiguring the SageMaker training job...")
sklearn_estimator = SKLearn(
    entry_point="train.py",    # name of script inside the source_dir
    source_dir="src",          # folder that contains train.py & setup.py
    role=role_arn,
    instance_count=1,
    instance_type="ml.m5.large",
    framework_version="1.2-1",
    py_version="py3",
    sagemaker_session=sagemaker_session,
    output_path=f"s3://{bucket_name}/{prefix}/output",
    environment={"MONGO_DB_URL": MONGODB_URL},
)


# 3. Launch the Training Job
# This will package your 'src' directory, upload it to S3, and start the training instance.
print("\nStarting the training job. This may take several minutes...")
sklearn_estimator.fit()
print("\nTraining job completed successfully!")

# 4. Deploy the Trained Model to an Endpoint
# This creates a real-time HTTPS endpoint using your inference.py script.
print("\nDeploying the model to a SageMaker endpoint...")
predictor = sklearn_estimator.deploy(
    initial_instance_count=1,
    instance_type="ml.t2.medium"  # A cost-effective instance type for inference
)
print(f"\nEndpoint '{predictor.endpoint_name}' deployed successfully.")
print(f"Endpoint URL: {predictor.endpoint_name}")


# 5. Invoke the Endpoint for a Prediction
print("\nInvoking the endpoint with sample data...")
# This sample data must match the feature names and types your model expects.
sample_data = {
    "Gender": ["Male"], "Age": [30], "Driving_License": [1], "Region_Code": [28.0],
    "Previously_Insured": [0], "Vehicle_Age": ["1-2 Year"], "Vehicle_Damage": ["Yes"],
    "Annual_Premium": [35000.0], "Policy_Sales_Channel": [124.0], "Vintage": [150]
}

try:
    # The predictor handles JSON serialization automatically.
    prediction_result = predictor.predict(sample_data)
    print(f"\nPrediction successful! Result: {prediction_result}")
except Exception as e:
    print(f"\nError during prediction: {e}")


# 6. Clean Up Resources (IMPORTANT for Cost Management)
# Delete the endpoint to avoid ongoing charges.
print("\n--- Cleanup ---")
print("Would you like to delete the endpoint now? (yes/no)")
if input().lower() == 'yes':
    predictor.delete_endpoint()
    print(f"Endpoint '{predictor.endpoint_name}' has been deleted.")
else:
    print(f"Cleanup skipped. Please remember to delete the endpoint '{predictor.endpoint_name}' manually from the SageMaker console to avoid charges.")
