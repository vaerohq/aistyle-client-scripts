import argparse
import requests
import httpx
import asyncio
import os
from dotenv import load_dotenv
import sys
import readline
import nltk
import traceback
import time
import json
import tiktoken

def get_multiline_input(prompt="Enter text (end with 'END'):", end_keyword="END"):
    print(prompt)
    lines = []
    while True:
        line = input()
        if line.strip().upper() == end_keyword:
            break
        lines.append(line)
    return "\n".join(lines)

async def fetch_sse(endpoint, payload, headers):
    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:

        TIME_MODEL = True

        if TIME_MODEL:
            # initialize timing variables
            first_token_time = None
            total_tokens = 0
            total_content = ""
            elapsed_time = 0

            start_time = time.time()
        
        total_output = ""

        async with client.stream("POST", endpoint, json=payload, headers=headers) as response:
            if response.is_error:
                await response.aread() # read the error response body
                response.raise_for_status()  # Check if the response status is OK

            # Process the streaming response line by line
            async for line in response.aiter_raw():
                if TIME_MODEL and first_token_time is None:
                    first_token_time = time.time()  # Time when the first chunk is received
                    elapsed_time = first_token_time - start_time

                content = line.decode()
                
                print(content, end="", flush=True)
                total_output += content

        if TIME_MODEL:
            # end time
            end_time = time.time()

            # time to first token
            print(f"Time to first token: {elapsed_time:.4f} seconds")

            # tokens per second
            enc = tiktoken.encoding_for_model("gpt-4o")
            tokens = enc.encode(total_output)
    
            total_tokens = len(tokens)
            total_time_s = end_time - start_time
            tokens_per_second = total_tokens / total_time_s
            print(f"Total tokens: {total_tokens}")
            print(f"Total time: {total_time_s:.4f} seconds")
            print(f"Tokens per second: {tokens_per_second:.2f}")
        
        # Process and output the accumulated content
        print("\n----- PROCESSED OUTPUT -----")
        condensed_lines = []
        analytics_string = ""
        for line in total_output.splitlines():
            if line.startswith("Original: ") or line.startswith("Levenshtein Distance: ") or line.startswith("Levenshtein Ratio: ") or line.startswith("Length ratio: "):
                continue
            elif line.startswith("quality_analytics: "):
                analytics_string = line[len("quality_analytics: "):]
                continue
            elif line.startswith("Styled: "):
                condensed_lines.append(line[8:])
            else:         
                condensed_lines.append(line)
        
        condensed_lines = [ln for i, ln in enumerate(condensed_lines)
                    if ln.strip() or i == 0 or condensed_lines[i-1].strip()] # reduce consecutive empty lines to one

        for line in condensed_lines:
            print(line)
        
        # Pretty print the quality analytics
        if analytics_string:
            try:
                quality_data = json.loads(analytics_string)
                print("\n----- QUALITY ANALYTICS -----")
                print(json.dumps(quality_data, indent=2))
            except json.JSONDecodeError as e:
                print(f"\n----- QUALITY ANALYTICS (Raw - JSON Parse Error) -----")
                print(f"Error: {e}")
                print(analytics_string)


def main():
    # load environment variables
    load_dotenv()

    parser = argparse.ArgumentParser(description='Send message to AI style API')
    parser.add_argument('--model', type=str, help="ID of model to use")
    args = parser.parse_args()

    args.port = None
    args.host = "https://vaeroapi.com"

    token = os.getenv('VAERO_KEY', None)

    headers = {
        "Authorization": f"Bearer {token}"
    }

    while True:
        # Prompt user to select an endpoint
        print()
        print("Select an endpoint:")
        print("1. Files (/files)")
        print("2. Train (/fine_tuning/jobs)")
        print("3. Inference (/style/transform)")
        choice = input("Enter 1, 2, 3: ").strip()

        if choice == "1":
            print("Select an action:")
            print("1. Upload a file")
            print("2. List files")
            print("3. Retrieve file info")
            print("4. Retrieve file content")
            print("5. Delete file")
            file_choice = input("Enter 1, 2, 3, 4, or 5: ").strip()

            if file_choice == "1":

                # Handle file upload
                print("File upload selected.")
                
                file_paths = input("Enter the file paths (separated by commas): ").strip().split(",")

                # Loop over each file path
                for file_path in file_paths:
                    file_path = file_path.strip()  # Remove any extra whitespace

                    # Open the file in binary mode
                    with open(file_path, 'rb') as file_to_upload:
                        # Prepare the files dictionary to be sent in the POST request
                        files = {
                            'file': file_to_upload
                            }

                        endpoint = f"{args.host}:{args.port}/v1/files" if args.port else f"{args.host}/v1/files"

                        try:
                            response = requests.post(endpoint, files=files, headers=headers)
                            response.raise_for_status()
                            print("Response headers:")
                            print(response.headers)
                            print("Response from the server:")
                            print(response.json())
                            print(f"File ID: {response.json()['id']}")
                        except requests.exceptions.RequestException as e:
                            print(f"Error: {e}")
                            if response.content:
                                print("Response headers:")
                                print(response.headers)
                                print("Server response:", response.content)
                
                continue # Skip the rest of the loop since this was handled separately
            elif file_choice == "2":
                print("List files selected.")
                endpoint = f"{args.host}:{args.port}/v1/files" if args.port else f"{args.host}/v1/files"

                try:
                    response = requests.get(endpoint, headers=headers)
                    response.raise_for_status()
                    print("Response headers:")
                    print(response.headers)
                    print("Response from the server:")
                    print(response.json())

                    print("\nFiles (formatted):")
                    for file in response.json()['data']:
                        print(f"ID: {file['id']}")
                        print(f"Filename: {file['filename']}")
                        print(f"Size: {file['bytes']} bytes")
                        print(f"Created: {file['created_at']}")
                        print(f"Purpose: {file['purpose']}")
                        print("-" * 50)
                except requests.exceptions.RequestException as e:
                    print(f"Error: {e}")
                    if response.content:
                        print("Response headers:")
                        print(response.headers)
                        print("Server response:", response.content)
            elif file_choice == "3":
                print("Retrieve file info selected.")
                file_id = input("Enter the file id: ").strip()
                endpoint = f"{args.host}:{args.port}/v1/files/{file_id}" if args.port else f"{args.host}/v1/files/{file_id}"

                try:
                    response = requests.get(endpoint, headers=headers)
                    response.raise_for_status()
                    print("Response headers:")
                    print(response.headers)
                    print("Response from the server:")
                    print(response.json())
                    print(f"Filename: {response.json()['filename']}")
                except requests.exceptions.RequestException as e:
                    print(f"Error: {e}")
                    if response.content:
                        print("Response headers:")
                        print(response.headers)
                        print("Server response:", response.content)
            elif file_choice == "4":
                print("Retrieve file content selected.")
                file_id = input("Enter the file id: ").strip()
                endpoint = f"{args.host}:{args.port}/v1/files/{file_id}/content" if args.port else f"{args.host}/v1/files/{file_id}/content"

                try:
                    response = requests.get(endpoint, headers=headers)
                    response.raise_for_status()
                    print("Response headers:")
                    print(response.headers)
                    print("Response from the server:")
                    print(response.content)
                except requests.exceptions.RequestException as e:
                    print(f"Error: {e}")
                    if response.content:
                        print("Response headers:")
                        print(response.headers)
                        print("Server response:", response.content)
            elif file_choice == "5":
                print("Delete file selected.")
                file_id = input("Enter the file id: ").strip()
                endpoint = f"{args.host}:{args.port}/v1/files/{file_id}" if args.port else f"{args.host}/v1/files/{file_id}"

                try:
                    response = requests.delete(endpoint, headers=headers)
                    response.raise_for_status()
                    print("Response headers:")
                    print(response.headers)
                    print("Response from the server:")
                    print(response.json())
                    print(f"File id: {response.json()['id']}")
                except requests.exceptions.RequestException as e:
                    print(f"Error: {e}")
                    if response.content:
                        print("Response headers:")
                        print(response.headers)
                        print("Server response:", response.content)

        elif choice == "2":
            print("Select an action:")
            print("1. Train")
            print("2. List Fine-tuning Jobs")
            print("3. Get Fine-tuning Job")

            ftj_choice = input("Enter 1, 2, 3, 4, or 5: ").strip()

            if ftj_choice == "1":
                # Handle training
                print("Training selected.")

                #model = input("Enter the model: ").strip()
                training_files = input("Enter training file IDs (comma-separated): ").strip().split(',')

                # Perform strip() on each element of training_files
                training_files = [file_id.strip() for file_id in training_files]

                suffix = input("Enter a suffix (optional, up to 16 characters, press Enter to skip): ").strip()

                payload = {
                    "training_files": training_files,
                }

                if suffix:
                    payload["suffix"] = suffix

                endpoint = f"{args.host}:{args.port}/v1/fine_tuning/jobs" if args.port else f"{args.host}/v1/fine_tuning/jobs"

                # Send the request for the chosen endpoint
                try:
                    response = requests.post(endpoint, json=payload, headers=headers)
                    response.raise_for_status()
                    print("Response headers:")
                    print(response.headers)
                    print("Response from the server:")
                    print(response.json())
                    print(f"Fine-tuning job ID: {response.json()['id']}")
                except requests.exceptions.RequestException as e:
                    print(f"Error: {e}")
                    if response.content:
                        print("Response headers:")
                        print(response.headers)
                        print("Server response:", response.content)
            elif ftj_choice == "2":

                print("List Fine-tuning Jobs selected.")

                endpoint = f"{args.host}:{args.port}/v1/fine_tuning/jobs" if args.port else f"{args.host}/v1/fine_tuning/jobs"

                # Send the request for the chosen endpoint
                try:
                    response = requests.get(endpoint, headers=headers)
                    response.raise_for_status()
                    print("Response headers:")
                    print(response.headers)
                    print("Response from the server:")
                    print(response.json())
                    print(f"Fine-tuning jobs data: {response.json()['data']}")
                except requests.exceptions.RequestException as e:
                    print(f"Error: {e}")
                    if response.content:
                        print("Response headers:")
                        print(response.headers)
                        print("Server response:", response.content)

            elif ftj_choice == "3":

                print("Get Fine-tuning Job selected.")
                fine_tuning_job_id = input("Enter the fine tuning job id: ").strip()
                endpoint = f"{args.host}:{args.port}/v1/fine_tuning/jobs/{fine_tuning_job_id}" if args.port else f"{args.host}/v1/fine_tuning/jobs/{fine_tuning_job_id}"

                # Send the request for the chosen endpoint
                try:
                    response = requests.get(endpoint, headers=headers)
                    response.raise_for_status()
                    print("Response headers:")
                    print(response.headers)
                    print("Response from the server:")
                    print(response.json())
                    print(f"Fine-tuning jobs fine-tuned model: {response.json()['fine_tuned_model']} {response.json()['status']}")
                except requests.exceptions.RequestException as e:
                    print(f"Error: {e}")
                    if response.content:
                        print("Response headers:")
                        print(response.headers)
                        print("Server response:", response.content)
            
        elif choice == "3":
            # Handle inference
            print("Inference selected.")

            model = args.model

            mode = input("Select mode (options: 'full', 'rewrite') [default: 'full']: ").strip() or "full"

            stream = input("Stream y/n (optional, press Enter to skip): ").strip() or "n"
            if stream.lower() == "y":
                stream = True
            else:
                stream = False

            base_model = None
            base_model_service = None
            base_model_endpoint = None
            base_model_api_key = None
            base_temperature = None
            if mode == "full":

                base_model = input("Select base model (options: 'gpt-4o', 'gpt-5.2', 'gpt-5.4', 'claude-sonnet-4-5', 'claude-sonnet-4-6', 'claude-opus-4-5', 'claude-opus-4-6', 'claude-opus-4-7'): ").strip()

                base_model_service = input("Select base model service (options: 'auto', 'openai', 'azure', 'anthropic', 'fireworks', default is 'auto'): ").strip() or "auto"

                if base_model_service == "azure":
                    base_model_endpoint = input("Enter base model endpoint (optional, press Enter to skip): ").strip()
                else:
                    base_model_endpoint = None
                
                base_model_api_key = input("Enter base model API key (optional, press Enter to skip): ").strip()

                base_temperature = input("Enter base model temperature (optional, press Enter to skip): ").strip()

            include_original = input("Include original text comparison y/n (default is y): ").strip() or "y"
            include_distance = input("Include distance y/n (default is y): ").strip() or "y"
            include_quality = input("Include quality analytics y/n (default is y):").strip() or "y"
            include_rouge = input("Include rouge score y/n (default is y):").strip() or "y"

            skip_headings = input("Skip headings y/n (default is n): ").strip() or "n"
            
            messages = []
            message = None
            if mode == "full":
                while True:
                    role = input("Enter message role (system/user/assistant) or 'done' to finish: ").strip().lower()
                    if role == 'done':
                        break
                    if role not in ['system', 'user', 'assistant']:
                        print("Invalid role. Please enter 'system', 'user', or 'assistant'.")
                        continue
                    content = get_multiline_input(f"Enter {role} message content:")
                    messages.append({"role": role, "content": content})
                    print(f"{role.capitalize()} message added.")

                if not messages:
                    print("No messages entered. Returning to main menu.")
                    continue

                print("Messages to be sent:")
                for msg in messages:
                    print(f"Role: {msg['role']}")
                    print(f"Content: {msg['content']}")
                    print("---")
            elif mode == "rewrite":
                message = get_multiline_input()
                base_prompt = input("Enter base prompt: ").strip()

            
            payload = {
                "mode": mode,
                "model": model
            }
            
            if mode == "full":
                payload["messages"] = messages
                payload["full_mode_options"] = {
                    "base_model": base_model
                }
            elif mode == "rewrite":
                payload["message"] = message
                payload["rewrite_mode_options"] = {
                    "base_prompt": base_prompt
                }
            
            if stream:
                payload["stream"] = True
            else:
                payload["stream"] = False

            if base_temperature:
                payload["full_mode_options"]["base_temperature"] = float(base_temperature)
            
            if base_model_service:
                payload["full_mode_options"]["base_model_service"] = base_model_service

            if base_model_endpoint:
                payload["full_mode_options"]["base_model_endpoint"] = base_model_endpoint

            if base_model_api_key:
                payload["full_mode_options"]["base_model_api_key"] = base_model_api_key
            
            if include_distance.lower() == 'y':
                payload["include_distance"] = True
                #payload["include_length"] = True

            if include_original.lower() == 'y':
                payload["include_compare_original"] = True
            
            if include_quality.lower() == 'y':
                payload["include_quality"] = True
            
            if include_rouge.lower() == 'y':
                payload["include_rouge"] = True

            if skip_headings == "y":
                payload["skip_headings"] = True

            endpoint = f"{args.host}:{args.port}/v1/style/transform" if args.port else f"{args.host}/v1/style/transform"

            if stream:
                # Send the request for the chosen endpoint and handle SSE
                try:
                    asyncio.run(fetch_sse(endpoint, payload, headers))
                except Exception as e:
                    print(f"Error: {e.response.text}")
            else:
                try:
                    response = requests.post(endpoint, json=payload, headers=headers)
                    response.raise_for_status()
                    print("Response headers:")
                    print(response.headers)
                    print("Response from the server:")
                    print(response.json())
                    print(f"{response.json()['choices'][0]['message']['content']}")
                except requests.exceptions.RequestException as e:
                    print(f"Error: {e}")
                    if response.content:
                        print("Response headers:")
                        print(response.headers)
                        print("Server response:", response.content)

        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")
        
if __name__ == "__main__":
    main()
