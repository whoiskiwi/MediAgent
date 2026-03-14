#!/bin/bash
# One-time AWS infrastructure setup for medi-agent.
# Run this ONCE from your local machine before the first deployment.
#
# Usage:
#   chmod +x scripts/setup_infra.sh
#   ./scripts/setup_infra.sh

set -e

# ── Config ────────────────────────────────────────────────────────────────────
REGION="us-west-2"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPO="medi-agent"
ECS_CLUSTER="medi-agent-cluster"
ECS_SERVICE="medi-agent-service"
LOG_GROUP="/ecs/medi-agent"
TASK_ROLE="mediAgentTaskRole"
EXEC_ROLE="ecsTaskExecutionRole"

echo "Account: $ACCOUNT_ID  |  Region: $REGION"

# ── 1. ECR repository ─────────────────────────────────────────────────────────
echo "[1/6] Creating ECR repository..."
aws ecr create-repository \
    --repository-name $ECR_REPO \
    --region $REGION \
    --image-scanning-configuration scanOnPush=true \
    2>/dev/null || echo "  (already exists)"

# ── 2. CloudWatch log group ───────────────────────────────────────────────────
echo "[2/6] Creating CloudWatch log group..."
aws logs create-log-group \
    --log-group-name $LOG_GROUP \
    --region $REGION \
    2>/dev/null || echo "  (already exists)"

# ── 3. ECS Task Role (app permissions: DynamoDB + SageMaker) ─────────────────
echo "[3/6] Creating ECS Task Role ($TASK_ROLE)..."
aws iam create-role \
    --role-name $TASK_ROLE \
    --assume-role-policy-document '{
        "Version":"2012-10-17",
        "Statement":[{
            "Effect":"Allow",
            "Principal":{"Service":"ecs-tasks.amazonaws.com"},
            "Action":"sts:AssumeRole"
        }]
    }' 2>/dev/null || echo "  (already exists)"

aws iam put-role-policy \
    --role-name $TASK_ROLE \
    --policy-name mediAgentPolicy \
    --policy-document "{
        \"Version\": \"2012-10-17\",
        \"Statement\": [
            {
                \"Effect\": \"Allow\",
                \"Action\": [
                    \"dynamodb:Query\",
                    \"dynamodb:GetItem\",
                    \"dynamodb:Scan\"
                ],
                \"Resource\": \"arn:aws:dynamodb:$REGION:$ACCOUNT_ID:table/DoctorSchedule\"
            },
            {
                \"Effect\": \"Allow\",
                \"Action\": \"sagemaker:InvokeEndpoint\",
                \"Resource\": [
                    \"arn:aws:sagemaker:$REGION:$ACCOUNT_ID:endpoint/medi-agent-classifier\",
                    \"arn:aws:sagemaker:$REGION:$ACCOUNT_ID:endpoint/medi-agent-generator\"
                ]
            }
        ]
    }"

# ── 4. ECS Task Execution Role (ECR pull + CloudWatch logs) ──────────────────
echo "[4/6] Creating ECS Task Execution Role ($EXEC_ROLE)..."
aws iam create-role \
    --role-name $EXEC_ROLE \
    --assume-role-policy-document '{
        "Version":"2012-10-17",
        "Statement":[{
            "Effect":"Allow",
            "Principal":{"Service":"ecs-tasks.amazonaws.com"},
            "Action":"sts:AssumeRole"
        }]
    }' 2>/dev/null || echo "  (already exists)"

aws iam attach-role-policy \
    --role-name $EXEC_ROLE \
    --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy \
    2>/dev/null || true

# ── 5. ECS Cluster ────────────────────────────────────────────────────────────
echo "[5/6] Creating ECS cluster ($ECS_CLUSTER)..."
aws ecs create-cluster \
    --cluster-name $ECS_CLUSTER \
    --capacity-providers FARGATE \
    --region $REGION \
    2>/dev/null || echo "  (already exists)"

# ── 6. Patch task-definition.json with real ACCOUNT_ID ───────────────────────
echo "[6/6] Patching infrastructure/task-definition.json..."
sed -i.bak "s/ACCOUNT_ID/$ACCOUNT_ID/g" infrastructure/task-definition.json
rm -f infrastructure/task-definition.json.bak
echo "  Done — task-definition.json updated with account $ACCOUNT_ID"

# ── Summary ───────────────────────────────────────────────────────────────────
ECR_URI="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$ECR_REPO"

echo ""
echo "✓ Infrastructure ready. Now set these GitHub Secrets:"
echo ""
echo "  Secret name              Value"
echo "  ─────────────────────── ─────────────────────────────────"
echo "  AWS_ACCESS_KEY_ID        (your IAM key)"
echo "  AWS_SECRET_ACCESS_KEY    (your IAM secret)"
echo ""
echo "  ECR URI (for reference): $ECR_URI"
echo ""
echo "Next steps:"
echo "  1. Push code to main branch → GitHub Actions will deploy automatically"
echo "  2. First deploy: ECS Service will be created by the workflow"
echo "  3. After first deploy, the API is at the ECS task's public IP:8000"
