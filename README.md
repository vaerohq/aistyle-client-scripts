# Vaero API Tester

A command-line tool for testing and interacting with the Vaero API for file management, model training, and AI-powered text style transformation.

## Usage

```bash
python vaero_api_tester.py [--model MODEL_ID]
```

### Parameters

- `--model MODEL_ID` - Optional. ID of the model to use for inference operations.

### Environment Variables

- `VAERO_KEY` - Required. Your Vaero API authentication token (can be set in a `.env` file).

### Interactive Features

The script provides an interactive menu for:
1. **Files** - Upload, list, retrieve info/content, and delete files
2. **Training** - Create fine-tuning jobs, list jobs, and get job status  
3. **Inference** - Transform text using AI style models with various options 