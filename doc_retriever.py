import json
import os

def get_qualitative_data(base_dir, dataset, model, doc_index):
    """
    Retrieves source text, target keyphrases, and predicted keyphrases
    for a specific document from KPEval-style output files.

    Args:
        base_dir (str): The path to the 'model_outputs' directory.
        dataset (str): The name of the dataset (e.g., 'kp20k').
        model (str): The name of the model folder (e.g., '1_tf-idf').
        doc_index (int): The 0-based index of the document to retrieve.

    Returns:
        dict: A dictionary containing 'source', 'target', and 'predictions'
              for the specified document, or None if not found or an error occurs.
    """
    # Construct the file path
    file_name = f"{dataset}_hypotheses_linked.json"
    file_path = os.path.join(base_dir, dataset, model, file_name)

    print(f"Attempting to read document index {doc_index} from: {file_path}")

    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return None

    line_found = False
    result_data = None

    # Open and read the JSON Lines file line by line
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i == doc_index:
                    # Found the correct line, parse the JSON
                    result_data = json.loads(line.strip())
                    line_found = True
                    break # Stop reading once the line is found

    except FileNotFoundError:
        print(f"Error: File not found during reading: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON on line {doc_index+1} in file {file_path}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while reading file {file_path}: {e}")
        return None


    if not line_found:
        print(f"Error: Document index {doc_index} not found in file {file_path}. File might be shorter than expected.")
        return None

    # Check if keys exist in the loaded data
    if not all(key in result_data for key in ['source', 'target', 'predictions']):
         print(f"Warning: Retrieved data for index {doc_index} might be missing expected keys ('source', 'target', 'predictions').")

    return result_data

# ---  Usage ---

project_root = "."
base_output_dir = os.path.join(project_root, "model_outputs")

dataset = 'kp20k'
model = '19_gpt3-five-shot'
doc_index = 569

retrieved_data = get_qualitative_data(base_output_dir, dataset, model, doc_index)

if retrieved_data:
    print("\n--- Retrieved Data ---")
    print(f"Source Text:\n{retrieved_data.get('source', 'N/A')}\n")
    print(f"Target Keyphrases:\n{retrieved_data.get('target', 'N/A')}\n")
    print(f"Predicted Keyphrases:\n{retrieved_data.get('predictions', 'N/A')}\n")
else:
    print(f"\nCould not retrieve data for the specified example.")
