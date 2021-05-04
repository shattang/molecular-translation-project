gcloud config set project deeplearning-311922

export IMAGE_FAMILY="pytorch-latest-gpu"
export ZONE="us-central1-f"
export INSTANCE_NAME="deepl3"
export MACHINE_TYPE="n1-standard-8"

gcloud compute instances create $INSTANCE_NAME \
  --zone=$ZONE \
  --machine-type=$MACHINE_TYPE \
  --image-family=$IMAGE_FAMILY \
  --image-project=deeplearning-platform-release \
  --maintenance-policy=TERMINATE \
  --accelerator="type=nvidia-tesla-t4,count=1" \
  --metadata="install-nvidia-driver=True"
  #--preemptible

# gcloud compute instances attach-disk $INSTANCE_NAME --disk disk-2 --zone $ZONE
