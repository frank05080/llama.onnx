#!/bin/bash

# Define the array with 50 sample questions
contexts=(
  "What is Nvidia?"
  "What is Horizon Robotics?"
  "How does machine learning work?"
  "What is artificial intelligence?"
  "What are neural networks?"
  "How does a convolutional neural network (CNN) function?"
  "What is deep learning?"
  "What is reinforcement learning?"
  "How do decision trees work?"
  "What are support vector machines (SVM)?"
  "What is natural language processing (NLP)?"
  "What is a chatbot?"
  "How do self-driving cars work?"
  "What is computer vision?"
  "What is image recognition?"
  "How does speech recognition work?"
  "What is a recommendation system?"
  "What is big data?"
  "How do algorithms work?"
  "What is the Internet of Things (IoT)?"
  "What is edge computing?"
  "What is cloud computing?"
  "How does blockchain technology work?"
  "What is cryptocurrency?"
  "How do quantum computers work?"
  "What is cybersecurity?"
  "What are autonomous systems?"
  "What is robotics?"
  "How do drones work?"
  "What is 5G technology?"
  "How do virtual reality (VR) systems work?"
  "What is augmented reality (AR)?"
  "What is mixed reality (MR)?"
  "What is the metaverse?"
  "What are digital twins?"
  "What is Industry 4.0?"
  "How does 3D printing work?"
  "What is additive manufacturing?"
  "What are smart cities?"
  "What is predictive analytics?"
  "What is data mining?"
  "What is data science?"
  "How does a data warehouse work?"
  "What is a data lake?"
  "What is machine learning ops (MLOps)?"
  "What is DevOps?"
  "How does a neural network train?"
  "What is transfer learning?"
  "What is federated learning?"
  "What is synthetic data?"
  # Add more contexts as needed
)


# Loop through the array with index
for i in "${!contexts[@]}"; do
  context=${contexts[$i]}
  echo "Running: tools/python3 onnx_RWKV_in_150_lines.py --context \"$context\" --input_folder_path \"/home/ros/share_dir/gitrepos/llama.onnx/data/ptdumped_inputs$i\" --dump_inputs True"
  python3 tools/onnx_RWKV_in_150_lines.py --context "$context" --input_folder_path "/home/ros/share_dir/gitrepos/llama.onnx/data/ptdumped_inputs$i" --dump_inputs True
done
