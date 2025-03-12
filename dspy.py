import os
import dspy
from datasets import load_dataset
from datetime import datetime

# 1. Prepare the Dataset
def prepare_qa_dataset():
    """Load and prepare a simple QA dataset."""
    dataset = load_dataset("squad", split="train")
    
    # Create smaller samples with proper Example objects
    train_data = []
    for example in dataset.select(range(100)):
        train_data.append(
            dspy.Example(
                context=example["context"],
                question=example["question"],
                answer=example["answers"]["text"][0]
            ).with_inputs("context", "question")
        )
    
    test_data = []
    for example in dataset.select(range(100, 150)):
        test_data.append(
            dspy.Example(
                context=example["context"],
                question=example["question"],
                answer=example["answers"]["text"][0]
            ).with_inputs("context", "question")
        )
    
    return train_data, test_data

# 2. Define the Problem
class QuestionAnswering(dspy.Signature):
    """Answer questions based on provided context."""
    context = dspy.InputField(desc="passage containing the answer")
    question = dspy.InputField(desc="question to be answered")
    answer = dspy.OutputField(desc="answer extracted from the context")

# 3. Define the Metric
def qa_metric(example, pred, trace=None):
    """Simple exact match metric for QA."""
    return int(example.answer.lower() == pred.answer.lower())

# 4. Baseline LM Program
class BaselineQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(QuestionAnswering)
    
    def forward(self, context, question):
        return self.predict(context=context, question=question)

def main():
    # Load dataset
    train_data, test_data = prepare_qa_dataset()
    
    # Configure LM
    # api_key = os.getenv("OPENAI_KEY")
    # llm = dspy.LM('openai/gpt-3.5-turbo', api_key=api_key)
    llm = dspy.LM('ollama_chat/gemma2', api_base='http://localhost:11434', api_key='')
    dspy.configure(lm=llm)
    
    # Create baseline model
    baseline = BaselineQA()
    
    # Evaluate baseline
    evaluator = dspy.Evaluate(
        devset=test_data,
        metric=qa_metric,
        num_threads=2,
        display_progress=True
    )
    baseline_score = evaluator(baseline)
    print(f"Baseline Score: {baseline_score:.2%}")
    
    # Use MIPROv2 for optimization
    optimizer = dspy.MIPROv2(metric=qa_metric)
    
    # Optimize with a small trainset
    optimized = optimizer.compile(
        baseline, 
        trainset=train_data,
        max_bootstrapped_demos=4,
        requires_permission_to_run=False
    )
    
    # Re-evaluate
    optimized_score = evaluator(optimized)
    print(f"Optimized Score: {optimized_score:.2%}")
    
    # Display API call for verification
    try:
        print("\nAPI Call Example:")
        call = llm.history[-1]
        print(f"Endpoint: {call.model}")
        print(f"Prompt: {call.prompt[:100]}...")
    except Exception as e:
        print(f"API Call Example: {str(e)}")

if __name__ == "__main__":
    main()
