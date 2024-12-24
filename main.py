from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os
import re
from bs4 import BeautifulSoup
from datasets import Dataset
import gc
import torch
import pandas as pd
from transformers import Trainer, TrainingArguments


# Text cleaning function
def clean_text(text):
    if isinstance(text, float):  # Handle NaN values
        return ""
    text = str(text).lower()  # Ensure text is a string
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Convert to lowercase and strip leading/trailing spaces
    text = text.lower().strip()
    return text

def remove_html_tags(text):
    soup = BeautifulSoup(text, 'html.parser')
    return soup.get_text()

def remove_newlines(text):
    return text.replace('\n', ' ').replace('\r', ' ')



# Define the model directory
model_dir = 'G:/Year 4/NN/project/model weights'

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# Load the model for sequence classification
model = AutoModelForSequenceClassification.from_pretrained(model_dir, trust_remote_code=True)

print("Model and tokenizer loaded successfully!")


test = pd.read_csv('G:/Year 4/NN/project/test.csv')

# Apply preprocessing to the 'Discussion' column
test['Discussion'] = test['Discussion'].apply(clean_text)
test['Discussion'] = test['Discussion'].apply(remove_html_tags)
test['Discussion'] = test['Discussion'].apply(remove_newlines)

print("Preprocessing completed!")




# Tokenize the test data
test_encodings = tokenizer(test['Discussion'].tolist(), truncation=True, padding=True, max_length=128)

# Create a Hugging Face Dataset
test_dataset = Dataset.from_dict({
    'input_ids': test_encodings['input_ids'],
    'attention_mask': test_encodings['attention_mask']
})

# Free up memory
del test_encodings
gc.collect()
torch.cuda.empty_cache()

print("Test data tokenized!")




# Define a dummy Trainer for evaluation
training_args = TrainingArguments(
    output_dir="G:/Year 4/NN/project",
    per_device_eval_batch_size=16
)
trainer = Trainer(
    model=model,
    args=training_args
)

# Generate predictions
predictions = trainer.predict(test_dataset)
predicted_labels = predictions.predictions.argmax(axis=-1)  # Predicted class labels (encoded values)

print("Predictions generated!")





# Create submission dataframe
submission_df = pd.DataFrame({
    'SampleID': test['SampleID'],  # Ensure your test CSV has 'SampleID' column
    'Category': predicted_labels  # Use the encoded values directly
})

# Save submission file
submission_path = 'G:/Year 4/NN/project/sentence-transformers-all-mpnet-base-v2-submission80.csv'
submission_df.to_csv(submission_path, index=False)

print(f"Submission file saved to {submission_path}")