import modal

app = modal.App("grpo-train")

image = modal.Image.from_registry("pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime").env({
    "CUDA_HOME": "/usr/local/cuda",
    "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:/usr/local/cuda/compat:/opt/conda/lib",
    "CC": "/usr/bin/gcc"
}).pip_install(
    "trl[vllm]==0.24.0",
    "datasets==3.5.1",
    "transformers==4.57.1",
    "bitsandbytes==0.48.2",
    "textstat==0.7.10",
    "accelerate>=0.24.0",
    "peft>=0.7.0",
    "trackio==0.5.0",
    "huggingface_hub",
    "kagglehub",
)

image = image.run_commands([
    "apt-get update",
    "apt-get install -y build-essential",
    "pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu121 triton==3.2.0"
])

@app.function(image=image, secrets=[modal.Secret.from_name("HF_TOKEN")], gpu="A100-40GB", timeout=18000)
def main():
    # will run in modal's container
    from kagglehub import model_download, KaggleDatasetAdapter, dataset_download
    import os
    os.environ.setdefault("BITSANDBYTES_CUDA_VERSION", "121")
    import textstat
    from peft import LoraConfig
    import torch
    import trackio
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed, TrainingArguments, TrainerCallback
    from datasets import Dataset, load_dataset
    from trl import GRPOConfig, GRPOTrainer
    from huggingface_hub import HfApi

    # constants
    # dataset for GRPO
    KAGGLE_DATASET_PATH = "roydontay/grpo-medquad-samples"

    HF_USER = "Jiahao123"
    HF_REPO_ID = f"{HF_USER}/medilite-grpo-v1" # change to v1 for repo
    # best model that has been finetuned
    INITIAL_FINETUNED_MODEL_REPO = "Jiahao123/SmolLM2-Instruct-1.7B-Medilite-Best-Finetuned"

    TRACK_IO_RUN_NAME = "medilite-grpo-2" # change to new run
    TRACK_IO_SPACE_ID = f"{HF_USER}/MediLiteQA"
    TRACK_IO_PROJECT = "medilite-grpo"

    # check using GPU/CPU
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using CUDA GPU: {torch.cuda.get_device_name()}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    else:
        device = "cpu"

    # Download dataset from Kaggle
    dataset_path = dataset_download(KAGGLE_DATASET_PATH)

    # Load and verify dataset from downloaded path
    medquad = load_dataset("csv", data_files=os.path.join(dataset_path, "grpo_train.csv"))

    def format_grpo_dataset(example, question_col, answer_col):
        """Formats QA datasets into chat format"""
        formatted_message_pairs = []
        answers_list = []
        
        for question, answer in zip(example[question_col], example[answer_col]):
            prompts = [
                {"role": "user", "content": question},
            ]
            answers = [answer]
            formatted_message_pairs.append(prompts)
            answers_list.append(answers)

        return {"prompt": formatted_message_pairs, "reference": answers_list}

    medquad_formatted = medquad['train'].map(
        format_grpo_dataset,
        batched=True,
        fn_kwargs={"question_col": "question", "answer_col": "answer"},
        remove_columns=['question', 'answer', 'source', 'focus_area']
    )

    medquad_train_formatted = medquad_formatted.select(range(0, 4921, 1))
    medquad_eval_formatted = medquad_formatted.select(range(4921, len(medquad_formatted), 1))

    print("Number of training samples:", medquad_train_formatted.num_rows)
    print("Number of evaluation samples:", medquad_eval_formatted.num_rows)

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        INITIAL_FINETUNED_MODEL_REPO,
        device_map="auto",
        trust_remote_code=True
    )

    # Reward functions
    def reward_readability(completions, **kwargs):
        """Reward function that rewards completions with higher readability score."""
        completion_contents = [completion[0]["content"] for completion in completions]
        rewards = []
        for completion in completion_contents:
            score = textstat.flesch_reading_ease(completion)
            score = max(0.0, min(score, 100.0))
            scaled = (score / 50.0) - 1.0
            rewards.append(float(scaled))
        return rewards
    
    def soft_length_reward(length, lower=30, upper=300, slope=0.02):
        """
        Returns a reward based on word length.
        - Reward = 1.0 for 30 <= length <= 300
        - Soft linear decay outside bounds, down to -1.0
        - 'slope' controls how quickly the punishment reaches -1.0
        """
        if lower <= length <= upper:
            return 1.0
        elif length < lower:
            # Decay linearly from -1 to 1 as length goes from 0 → lower
            reward = -1.0 + (length / lower) * 2.0
            return max(-1.0, min(1.0, reward))
        else:
            # Decay linearly from 1 to -1 as length goes from upper → upper + 1/slope range
            reward = 1.0 - (length - upper) * slope
            return max(-1.0, min(1.0, reward))    
    
    def reward_ideal_length(completions, **kwargs):
        completion_contents = [completion[0]["content"] for completion in completions]
        rewards = []
        for completion in completion_contents:
            score = soft_length_reward(len(completion.split()), lower=30, upper=3000, slope=0.02)
            rewards.append(float(score))
        return rewards

    training_args = GRPOConfig(
        # Essential parameters
        output_dir="medilite-grpo-lora",
        num_train_epochs=3, # choose 3 epochs for training
        num_generations=4,  # Number of completions to generate for each prompt
        per_device_train_batch_size=4,  # We want to get all generations in one device batch
        loss_type="dapo",
        gradient_accumulation_steps=8,
        learning_rate=5e-5,
        beta=0.2,
        repetition_penalty=1.5,

        logging_strategy="steps",
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
    
        # Hugging Face Hub integration
        push_to_hub=True,  # Set to True to upload to Hub
        hub_model_id=HF_REPO_ID,
    
        # Experiment tracking
        report_to=["trackio"],
        run_name=TRACK_IO_RUN_NAME,
        trackio_space_id=TRACK_IO_SPACE_ID,
        project=TRACK_IO_PROJECT,

        # use vllm for training
        use_vllm=True,
        vllm_mode="colocate",
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_readability, reward_ideal_length],
        args=training_args,
        peft_config=peft_config,
        train_dataset=medquad_train_formatted,
        eval_dataset=medquad_eval_formatted,
    )

    trainer.train()
    trainer.push_to_hub()

if __name__ == "__main__":
    with app.run():
        main.remote()